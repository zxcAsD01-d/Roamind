"""YOLO keypoints + same-frame stereo, without simulator bone positions."""
from __future__ import annotations

import math

import numpy as np

NAMES = ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
         'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
         'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
         'left_knee', 'right_knee', 'left_ankle', 'right_ankle')


def decode_poses(prediction):
    if prediction.keypoints is None:
        return []
    output = []
    for box, points in zip(prediction.boxes, prediction.keypoints.data.cpu().numpy()):
        output.append({'box': box.xyxy[0].tolist(), 'confidence': float(box.conf.item()),
                       'keypoints': [{'name': name, 'pixel': point[:2].round(2).tolist(),
                                     'confidence': round(float(point[2]), 3)} for name, point in zip(NAMES, points)]})
    return output


def metric_keypoint(frame, keypoint, pose=None):
    pixel, confidence = keypoint.get('pixel', []), keypoint.get('confidence', 0)
    if len(pixel) != 2 or not np.isfinite(pixel).all() or confidence < .4:
        return {**keypoint, 'valid': False, 'reason': '关键点置信度不足'}
    x, y = np.rint(pixel).astype(int)
    height, width = frame.depth.shape
    x0, x1, y0, y1 = max(0, x-5), min(width, x+6), max(0, y-5), min(height, y+6)
    if x0 >= x1 or y0 >= y1:
        return {**keypoint, 'valid': False, 'reason': '关键点不在画面内'}
    depth = frame.depth[y0:y1, x0:x1]
    values = depth[frame.valid[y0:y1, x0:x1]]
    if len(values) < 8:
        return {**keypoint, 'valid': False, 'reason': '关键点周围双目深度不足'}
    bins = np.floor(values/.08).astype(int)
    peak = np.bincount(bins).argmax()
    cluster = values[np.abs(bins-peak) <= 1]
    z = float(np.median(cluster))
    spread = float(np.percentile(cluster, 90)-np.percentile(cluster, 10))
    if len(cluster) < 8 or spread > .18:
        return {**keypoint, 'valid': False, 'reason': '深度边缘或遮挡不稳定'}
    c = frame.calibration
    optical = np.array([(pixel[0]-c.cx)*z/c.fx, (pixel[1]-c.cy)*z/c.fy, z])
    result = {**keypoint, 'valid': True, 'opticalPosition': optical.round(4).tolist(),
              'distance': round(float(np.linalg.norm(optical)), 4), 'depth': round(z, 4),
              'depthSpread': round(spread, 4), 'samples': len(cluster),
              'sigmaMeters': round(max(.015, z*z/(c.fx*c.baseline)*.6, spread/2), 4)}
    if pose is not None:
        result['worldPosition'] = (pose[:3, :3]@optical+pose[:3, 3]).round(4).tolist()
    return result


def fuse_people(frame, people, pose, observed_at):
    result = []
    for person in people[:8]:
        points = [metric_keypoint(frame, point, pose) for point in person['keypoints']]
        valid = [p['opticalPosition'] for p in points if p.get('valid')]
        center = np.median(valid, axis=0).tolist() if valid else None
        result.append({**person, 'keypoints': points, 'opticalPosition': center,
                       'observedAt': observed_at, 'source': 'yolo11-pose+stereo-sgbm-lr',
                       'distance': None if center is None else round(math.sqrt(sum(v*v for v in center)), 4)})
    return result
