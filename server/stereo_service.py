"""Stereo sessions: depth, visual pose and time-aligned semantic fusion."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

import cv2
import numpy as np

from semantic_map import SemanticMap
from stereo_depth import Calibration, StereoDepth, clearance_details, decode_image, depth_preview
from visual_odometry import VisualOdometry
from navigation_grid import navigation_grid
from floor_tracking import FloorTracker
from pointcloud_exits import PointCloudExits
from touch_vision import BodyTargets, read_rig


def visual_body_position(raw_rig, pose):
    raw = raw_rig.get('bodyInOptical') if isinstance(raw_rig, dict) else None
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError('自身身体到相机的标定需要三个有限坐标')
    point = np.asarray(raw, dtype=float)
    if not np.isfinite(point).all() or np.linalg.norm(point) > 5:
        raise ValueError('自身身体到相机的标定无效')
    return (pose[:3, :3]@point+pose[:3, 3]).round(5).tolist()


class StereoSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.depth = StereoDepth()
        self.odometry = VisualOdometry()
        self.world = SemanticMap()
        self.frames = OrderedDict()
        self.annotations = {}
        self.last_seen = time.monotonic()
        self.last_frame = -1
        self.calibration = None
        self.latest = None
        self.rigs = OrderedDict()
        self.body_targets = BodyTargets()
        self.floor = FloorTracker()
        self.exits = PointCloudExits()

    def infer(self, body):
        started = time.perf_counter()
        timestamp, frame_id = float(body.get('observedAt', 0)), body.get('frameId')
        if not np.isfinite(timestamp) or abs(time.time()-timestamp) > 2:
            raise ValueError('双目帧过期或时间戳无效')
        if type(frame_id) is not int or frame_id <= self.last_frame:
            raise ValueError('双目 frameId 必须递增，旧帧不可覆盖新地图')
        left, right = decode_image(body.get('left')), decode_image(body.get('right'))
        calibration = Calibration.read(body.get('calibration'), left.shape[:2])
        rig = read_rig(body.get('rig'))
        if self.calibration is not None and self.calibration != calibration:
            raise ValueError('双目标定改变，请重置视觉地图')
        self.calibration, self.last_frame = calibration, frame_id
        frame = self.depth.infer(left, right, calibration)
        pose = self.odometry.update(frame, timestamp, rig)
        tracked = pose['status'] in ('initialized', 'tracking', 'relocalized')
        body_position = visual_body_position(body.get('rig'), self.odometry.pose)
        if body_position is not None:
            pose['bodyPosition'] = body_position
        if tracked:
            self.world.integrate(frame, self.odometry.pose, timestamp)
        self.frames[frame_id] = (frame, self.odometry.pose.copy() if tracked else None, timestamp)
        self.rigs[frame_id] = rig
        while len(self.frames) > 8:
            old, _ = self.frames.popitem(last=False)
            self.rigs.pop(old, None)
            self.annotations.pop(old, None)
        prior = self.floor.control_prior(self.odometry.pose, rig) if tracked else None
        sectors, plane, ground = clearance_details(frame, rig, prior)
        rig_body = body.get('rig')
        foot = rig_body.get('bodyInOptical') if isinstance(rig_body, dict) else None
        if ground is None and tracked:
            # VO height can drift. Refit only current depth close to the
            # calibrated soles; never promote a remembered plane to free floor.
            candidate = self.floor.body_prior(rig, foot)
            if candidate is not None:
                sectors, plane, ground = clearance_details(frame, rig, candidate)
        if tracked:
            self.floor.update(ground, self.odometry.pose, rig, foot)
        exit_evidence = (self.exits.analyze(self.world, self.floor, self.odometry.pose, timestamp, calibration, ground)
                         if tracked else None)
        self.last_seen = time.monotonic()
        grid = cv2.resize(frame.depth, (80, 45), interpolation=cv2.INTER_NEAREST)
        result = {'frameId': frame_id, 'observedAt': timestamp, 'method': 'stereo-sgbm-lr',
                'validRatio': round(float(frame.valid.mean()), 4), 'sectors': sectors,
                'surfaceCompletion': frame.completion_info,
                'groundPlane': plane, 'groundInfo': None if ground is None else {
                    'normal': ground['normal'].round(5).tolist(), 'offset': round(float(ground['offset']), 5),
                    'support': ground['support'], 'residual': round(ground['residual'], 5),
                    'tiltDeg': round(ground['tiltDeg'], 3), 'frame': ground['frame']},
                'depthGrid': {'width': 80, 'height': 45, 'meters': grid.round(3).ravel().tolist()},
                'preview': depth_preview(frame), 'pose': pose, 'map': self.world.snapshot(timestamp),
                'trajectory': self.odometry.trajectory, 'baseline': calibration.baseline,
                'inferenceMs': round((time.perf_counter()-started)*1000, 2),
                'mapping': tracked, 'source': 'rgb-stereo-only',
                'navigationGrid': navigation_grid(frame, self.odometry.pose, rig, ground) if tracked else None,
                'exitEvidence': exit_evidence,
                'rig': None if rig is None else {'source': 'ego-kinematics', 'opticalToControl': rig.ravel().tolist()}}
        self.latest = result
        return result

    def annotate(self, body, detections):
        record = self.frames.get(body.get('frameId'))
        if record is None:
            return {'fused': False, 'reason': '双目帧已过期或未建立'}
        frame, pose, timestamp = record
        if pose is None or time.time()-timestamp > 2:
            return {'fused': False, 'reason': '源帧定位失效或语义结果过期'}
        frame_id = body['frameId']
        if frame_id not in self.annotations:
            self.annotations[frame_id] = self.world.annotate(frame, pose, detections, timestamp)
        associations = self.annotations[frame_id]
        from interaction_nodes import extract_nodes
        for association in associations:
            if 'nodes' not in association:
                association['nodes'] = extract_nodes(frame, pose, self.rigs.get(frame_id), association)
        snapshot = self.world.snapshot()
        return {'fused': True, 'frameId': body['frameId'], 'associations': associations,
                'mapRevision': snapshot['revision'], 'map': snapshot}

    def snapshot(self):
        with self.lock:
            latest = self.latest
            if latest is None:
                return {'ready': False, 'reason': '尚未收到双目帧', 'clientId': None}
            age = max(0., time.time()-latest['observedAt'])
            return {'ready': True, 'stale': age > 2., 'ageSeconds': round(age, 3),
                    'frameId': latest['frameId'], 'observedAt': latest['observedAt'],
                    'pose': self.odometry.snapshot(), 'trajectory': self.odometry.trajectory,
                    'map': self.world.snapshot(), 'sectors': latest['sectors'],
                    'groundPlane': latest['groundPlane'], 'mapping': latest['mapping'],
                    'groundInfo': latest.get('groundInfo'),
                    'exitEvidence': latest.get('exitEvidence'),
                    'bodyObservation': self.body_targets.snapshot(),
                    'source': 'rgb-stereo-only'}

    def pose_points(self, body, people):
        from pose_depth import fuse_people
        record = self.frames.get(body.get('frameId'))
        if record is None or time.time()-record[2] > 2:
            return []
        frame, pose, timestamp = record
        fused = fuse_people(frame, people, pose, timestamp)
        return self.body_targets.update(fused, frame, pose, timestamp, body['frameId'], self.rigs.get(body['frameId']))


class StereoSessions:
    def __init__(self):
        self.lock = threading.Lock()
        self.sessions = {}
        self.compute = threading.BoundedSemaphore(1)

    def session(self, body):
        client = body.get('clientId')
        if not isinstance(client, str) or not 1 <= len(client) <= 100:
            raise ValueError('双目 clientId 无效')
        with self.lock:
            expired = [key for key, value in self.sessions.items() if time.monotonic()-value.last_seen > 120]
            for key in expired:
                del self.sessions[key]
            if client not in self.sessions:
                if len(self.sessions) >= 4:
                    raise BlockingIOError('双目会话已满，请关闭不用的视觉窗口')
                self.sessions[client] = StereoSession()
            return self.sessions[client]

    def infer(self, body):
        session = self.session(body)
        if not self.compute.acquire(blocking=False):
            raise BlockingIOError('双目计算繁忙，请丢弃本帧，避免多窗口积压旧图像')
        if not session.lock.acquire(blocking=False):
            self.compute.release()
            raise BlockingIOError('双目正在处理上一帧，请丢弃旧帧')
        try:
            return session.infer(body)
        finally:
            session.lock.release()
            self.compute.release()

    def annotate(self, body, detections):
        session = self.session(body)
        with session.lock:
            return session.annotate(body, detections)

    def reset(self, body):
        session = self.session(body)
        with self.lock, session.lock:
            self.sessions[body['clientId']] = StereoSession()
        return {'reset': True, 'frame': 'next-valid-stereo-camera'}

    def release(self, body):
        client = body.get('clientId')
        if not isinstance(client, str) or not 1 <= len(client) <= 100:
            raise ValueError('双目 clientId 无效')
        with self.lock:
            removed = self.sessions.pop(client, None) is not None
        return {'released': removed}

    def pose_points(self, body, people):
        session = self.session(body)
        with session.lock:
            return session.pose_points(body, people)

    def snapshot(self, client_id):
        with self.lock:
            session = self.sessions.get(client_id)
        if session is None:
            raise ValueError('未找到该双目会话；请先在网页开启双目视觉')
        return session.snapshot()

    def touch_target(self, body):
        with self.lock:
            session = self.sessions.get(body.get('clientId'))
        if session is None:
            raise ValueError('未找到该视觉会话')
        with session.lock:
            return session.body_targets.resolve(body)

    def body_frame(self, client_id):
        with self.lock:
            session = self.sessions.get(client_id)
        if session is None:
            raise ValueError('未找到该视觉会话')
        with session.lock:
            return session.body_targets.snapshot(include_image=True)

    def capture_record(self, client_id, frame_id=None):
        with self.lock:
            session = self.sessions.get(client_id)
        if session is None:
            raise ValueError('请先开启角色双目视觉')
        with session.lock:
            frame_id = session.last_frame if frame_id is None else frame_id
            record = session.frames.get(frame_id)
            if record is None or not 0 <= time.time()-record[2] < 1.5:
                raise ValueError('需要新鲜且同帧的双目图像')
            frame, _, timestamp = record
            rig = session.rigs.get(frame_id)
            return frame, timestamp, frame_id, None if rig is None else rig.copy()

    def listing(self):
        with self.lock:
            return [{'clientId': key, 'lastFrame': session.last_frame,
                     'ageSeconds': round(time.monotonic()-session.last_seen, 2)}
                    for key, session in self.sessions.items()]


STEREO = StereoSessions()
