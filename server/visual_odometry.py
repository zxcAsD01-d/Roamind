"""Metric stereo visual odometry with bounded keyframe relocalization."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class Keyframe:
    pixels: np.ndarray
    descriptors: np.ndarray
    points: np.ndarray
    pose: np.ndarray
    timestamp: float
    gray: np.ndarray
    signature: np.ndarray


class VisualOdometry:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1400, fastThreshold=10, edgeThreshold=19)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.pose = np.diag([1., -1., -1., 1.])
        self.keys = []
        self.status = 'initializing'
        self.inliers = 0
        self.error = None
        self.last_timestamp = 0.
        self.trajectory = []
        self.gravity_tilt = 0.
        self.gravity_aligned = False
        self.reference = None
        self.flow_reference = None
        self.rig = None

    def _features(self, frame):
        mask = cv2.erode(frame.valid.astype(np.uint8)*255, np.ones((3, 3), np.uint8))
        keypoints, descriptors = self.orb.detectAndCompute(frame.gray, mask)
        if descriptors is None:
            return np.empty((0, 2)), None
        pixels = np.array([key.pt for key in keypoints], np.float32)
        return pixels, descriptors

    @staticmethod
    def _points(frame, pixels):
        coordinates = np.rint(pixels).astype(int)
        depths = frame.depth[coordinates[:, 1], coordinates[:, 0]]
        c = frame.calibration
        return np.column_stack(((pixels[:, 0]-c.cx)*depths/c.fx,
                                (pixels[:, 1]-c.cy)*depths/c.fy, depths))

    @staticmethod
    def _signature(descriptors):
        return np.unpackbits(descriptors, axis=1).mean(axis=0)

    def _candidates(self, descriptors):
        signature = self._signature(descriptors)
        recent = self.keys[-4:][::-1]
        older = sorted(self.keys[:-4], key=lambda key: np.linalg.norm(key.signature-signature))
        return recent+older[:12]

    def _estimate(self, key, pixels, descriptors, matrix, timestamp):
        matches = self.matcher.knnMatch(key.descriptors, descriptors, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .73*pair[1].distance
                and key.points[pair[0].queryIdx, 2] > .25]
        if len(good) < 15:
            return None
        source = np.array([key.points[m.queryIdx] for m in good], np.float32)
        target = np.array([pixels[m.trainIdx] for m in good], np.float32)
        return self._solve(key, source, target, matrix, timestamp)

    def _track(self, key, frame, timestamp):
        source = key.pixels.astype(np.float32).reshape(-1, 1, 2)
        options = {'winSize': (21, 21), 'maxLevel': 3,
                   'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)}
        target, valid, _ = cv2.calcOpticalFlowPyrLK(key.gray, frame.gray, source, None, **options)
        if target is None or valid is None:
            return None
        reverse, back_valid, _ = cv2.calcOpticalFlowPyrLK(frame.gray, key.gray, target, None, **options)
        if reverse is None or back_valid is None:
            return None
        consistent = np.linalg.norm((source-reverse).reshape(-1, 2), axis=1) < .8
        keep = valid.ravel().astype(bool) & back_valid.ravel().astype(bool) & consistent
        keep &= key.points[:, 2] > .25
        if np.sum(keep) < 15:
            return None
        # Pixel tracks remain useful even when this frame's metric PnP is
        # degenerate. Keep their original 3-D coordinates and original pose.
        self.flow_reference = Keyframe(target.reshape(-1, 2)[keep], None, key.points[keep],
                                       key.pose, key.timestamp, frame.gray.copy(), None)
        return self._solve(key, key.points[keep].astype(np.float32),
                           target.reshape(-1, 2)[keep], frame.calibration.matrix(), timestamp)

    def _gravity_refine(self, key, source, target, matrix, transform):
        pose, _ = self._level_pose(key.pose@np.linalg.inv(transform), self.rig)
        world = source@key.pose[:3, :3].T+key.pose[:3, 3]

        def rotation_at(yaw):
            return cv2.Rodrigues(np.array([0., yaw, 0.]))[0]@pose[:3, :3]

        def residual(parameters):
            optical = (world-parameters[1:])@rotation_at(parameters[0])
            projection = optical@matrix.T
            pixels = projection[:, :2]/np.maximum(projection[:, 2:], .01)
            return (pixels-target).ravel()

        fit = least_squares(residual, np.r_[0., pose[:3, 3]], loss='huber', f_scale=.75,
                            max_nfev=20)
        if not fit.success or not np.isfinite(fit.x).all():
            return None
        pose[:3, :3], pose[:3, 3] = rotation_at(fit.x[0]), fit.x[1:]
        return np.linalg.inv(pose)@key.pose

    def _solve(self, key, source, target, matrix, timestamp):
        ok, rotation, translation, inliers = cv2.solvePnPRansac(source, target, matrix, None,
            iterationsCount=100, reprojectionError=2., confidence=.995, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < 15 or len(inliers)/len(source) < .4:
            return None
        ids = inliers[:, 0]
        rotation, translation = cv2.solvePnPRefineLM(source[ids], target[ids], matrix, None, rotation, translation)
        transform = np.eye(4)
        transform[:3, :3] = cv2.Rodrigues(rotation)[0]
        transform[:3, 3] = translation.ravel()
        if self.rig is not None:
            transform = self._gravity_refine(key, source[ids], target[ids], matrix, transform)
            if transform is None:
                return None
            rotation = cv2.Rodrigues(transform[:3, :3])[0]
            translation = transform[:3, 3]
        dt = timestamp-key.timestamp
        if np.linalg.norm(translation) > min(3., .15+3*dt) or np.linalg.norm(rotation) > min(1.3, .1+3*dt):
            return None
        projected = cv2.projectPoints(source[ids], rotation, translation, matrix, None)[0].reshape(-1, 2)
        error = float(np.median(np.linalg.norm(projected-target[ids], axis=1)))
        if error > 1.5:
            return None
        tracked_points = source[ids]@transform[:3, :3].T+transform[:3, 3]
        return key.pose@np.linalg.inv(transform), len(ids), error, target[ids], tracked_points

    def _reference(self, frame, pixels, points, descriptors, timestamp, estimate):
        """Keep metric landmarks alive instead of retriangulating them each frame."""
        tracked_pixels, tracked_points = (np.empty((0, 2)), np.empty((0, 3))) if estimate is None else estimate[3:]
        mask = np.full(frame.gray.shape, 255, np.uint8)
        for pixel in tracked_pixels:
            cv2.circle(mask, tuple(np.rint(pixel).astype(int)), 6, 0, -1)
        coordinates = np.rint(pixels).astype(int)
        fresh = mask[coordinates[:, 1], coordinates[:, 0]] > 0
        all_pixels = np.vstack((tracked_pixels, pixels[fresh])).astype(np.float32)
        all_points = np.vstack((tracked_points, points[fresh]))
        if len(all_pixels) >= 15:
            self.reference = Keyframe(all_pixels, descriptors, all_points, self.pose.copy(), timestamp,
                                      frame.gray.copy(), None)

    @staticmethod
    def _level_pose(pose, rig):
        """用相机外参的重力方向消除视觉里程计的 roll/pitch 漂移。"""
        matrix = np.asarray(rig, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            return pose, 0.
        gravity_optical = matrix[:3, :3].T@np.array([0., 1., 0.])
        gravity_map = pose[:3, :3]@gravity_optical
        length = np.linalg.norm(gravity_map)
        if length < 1e-6:
            return pose, 0.
        gravity_map /= length
        target = np.array([0., 1., 0.])
        axis = np.cross(gravity_map, target)
        sine = np.linalg.norm(axis)
        cosine = float(np.clip(gravity_map@target, -1., 1.))
        angle = float(np.arctan2(sine, cosine))
        if angle < 1e-6:
            return pose, 0.
        if sine < 1e-6:
            axis = np.array([1., 0., 0.])
        else:
            axis /= sine
        correction = cv2.Rodrigues(axis*angle)[0]
        leveled = pose.copy()
        leveled[:3, :3] = correction@pose[:3, :3]
        return leveled, float(np.degrees(angle))

    def update(self, frame, timestamp, rig=None):
        self.rig = rig
        pixels, descriptors = self._features(frame)
        self.inliers, self.error = 0, None
        estimate = None
        if (descriptors is None or len(pixels) < 25) and not self.keys:
            self.status = 'lost' if self.keys else 'initializing'
            return self.snapshot()
        points = self._points(frame, pixels)
        if not self.keys:
            if np.sum(points[:, 2] > .25) < 25:
                return self.snapshot()
            self.status = 'initialized'
            self.pose, self.gravity_tilt = self._level_pose(self.pose, rig) if rig is not None else (self.pose, 0.)
        else:
            key = self.reference if self.reference is not None else self.keys[-1]
            self.flow_reference = None
            estimate = self._track(key, frame, timestamp)
            if estimate is None and descriptors is not None:
                for key in self._candidates(descriptors):
                    estimate = self._estimate(key, pixels, descriptors, frame.calibration.matrix(), timestamp)
                    if estimate is not None:
                        break
            if estimate is None:
                if self.flow_reference is not None:
                    self.reference = self.flow_reference
                self.status = 'lost'
                return self.snapshot()
            self.pose, self.inliers, self.error = estimate[:3]
            if rig is not None:
                self.pose, self.gravity_tilt = self._level_pose(self.pose, rig)
            self.status = 'tracking' if key is self.reference or key is self.keys[-1] else 'relocalized'
        self.gravity_aligned = rig is not None
        self._reference(frame, pixels, points, descriptors, timestamp, estimate)
        if descriptors is not None and np.sum(points[:, 2] > .25) >= 25 and self._needs_keyframe(timestamp):
            self.keys.append(Keyframe(pixels, descriptors, points, self.pose.copy(), timestamp,
                                     frame.gray.copy(), self._signature(descriptors)))
        self.keys = self.keys[-96:]
        self.last_timestamp = timestamp
        position = self.pose[:3, 3].round(4).tolist()
        if not self.trajectory or np.linalg.norm(np.asarray(position)-self.trajectory[-1]) > .03:
            self.trajectory.append(position)
            self.trajectory = self.trajectory[-512:]
        return self.snapshot()

    def _needs_keyframe(self, timestamp):
        if not self.keys:
            return True
        key = self.keys[-1]
        movement = np.linalg.norm(self.pose[:3, 3]-key.pose[:3, 3])
        rotation = cv2.Rodrigues(key.pose[:3, :3].T@self.pose[:3, :3])[0]
        return movement > .15 or np.linalg.norm(rotation) > .12 or timestamp-key.timestamp > 2.

    def snapshot(self):
        return {'status': self.status, 'position': self.pose[:3, 3].round(4).tolist(),
                'cameraToWorld': self.pose.round(6).ravel().tolist(), 'inliers': self.inliers,
                'reprojectionError': self.error, 'observedAt': self.last_timestamp,
                'frame': 'gravity-aligned-y-up', 'source': 'stereo-orb-pnp-ransac',
                'gravityAligned': self.gravity_aligned, 'tiltCorrectionDeg': round(self.gravity_tilt, 3)}
