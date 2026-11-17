"""Resolve body targets from time-aligned pose pixels and stereo surface samples."""
from __future__ import annotations

import copy
import base64
import time
import uuid
from collections import OrderedDict

import numpy as np
import cv2

from pose_depth import metric_keypoint

SOURCE = 'yolo11-pose+stereo-sgbm-lr'
LIMBS = [('shoulder', 'elbow'), ('elbow', 'wrist'), ('hip', 'knee'), ('knee', 'ankle')]
REGIONS = {'upper_arm': ('shoulder', 'elbow'), 'forearm': ('elbow', 'wrist'),
           'thigh': ('hip', 'knee'), 'shin': ('knee', 'ankle')}


def read_rig(raw):
    if not isinstance(raw, dict) or raw.get('source') != 'ego-kinematics':
        return None
    matrix = np.asarray(raw.get('opticalToControl', []), dtype=float)
    if matrix.size != 16 or not np.isfinite(matrix).all():
        raise ValueError('相机外参需要 16 个有限值')
    matrix = matrix.reshape(4, 4)
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-4)
            or abs(np.linalg.det(rotation)-1) > 1e-4):
        raise ValueError('相机外参必须为刚体变换；不接受缩放或反射')
    return matrix


def body_pixel(person, selection):
    points = {p['name']: p for p in person['keypoints'] if p['confidence'] >= .6}
    if selection.get('bodyPart'):
        region = region_selection(selection['bodyPart'])
        if region:
            return follow_selection(person, region), region
        point = points.get(selection['bodyPart'])
        if point is None:
            raise ValueError('该身体部位未清晰识别')
        return point, {'bodyPart': point['name'], 'estimate': 'detected-keypoint'}
    pixel = np.asarray(selection.get('pixel', []), dtype=float)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError('指定 bodyPart 或同帧画面 pixel')
    candidates = []
    for side in ('left', 'right'):
        for first, second in LIMBS:
            a, b = points.get(f'{side}_{first}'), points.get(f'{side}_{second}')
            if a is None or b is None:
                continue
            pa, pb = np.asarray(a['pixel']), np.asarray(b['pixel'])
            delta = pb-pa
            t = float(np.clip((pixel-pa)@delta/max(delta@delta, 1e-6), 0, 1))
            distance = np.linalg.norm(pixel-pa-t*delta)
            if distance <= max(6., .08*np.linalg.norm(delta)):
                relative = pixel-pa
                offset = float((delta[0]*relative[1]-delta[1]*relative[0])/(delta@delta)) if delta@delta > 1e-6 else 0.
                candidates.append((distance, a, b, t, offset))
    if not candidates:
        return face_pixel(person, pixel)
    _, a, b, t, offset = min(candidates, key=lambda item: item[0])
    selection = {'segment': [a['name'], b['name']], 'fraction': t, 'normalOffset': offset,
                 'estimate': 'inferred-limb-surface'}
    return follow_selection(person, selection), selection


def region_selection(name):
    for side in ('left', 'right'):
        region = REGIONS.get(name.removeprefix(side+'_')) if name.startswith(side+'_') else None
        if region:
            return {'segment': [side+'_'+part for part in region], 'fraction': .5,
                    'normalOffset': 0., 'bodyRegion': name, 'estimate': 'inferred-limb-region'}
    return None


def face_pixel(person, pixel):
    from itertools import combinations
    points = [p for p in person['keypoints'] if p['confidence'] >= .6
              and p['name'] in ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear')]
    for triangle in combinations(points, 3):
        vertices = np.asarray([p['pixel'] for p in triangle])
        matrix = np.vstack((vertices.T, np.ones(3)))
        if abs(np.linalg.det(matrix)) < 10:
            continue
        weights = np.linalg.solve(matrix, [*pixel, 1.])
        if np.min(weights) >= 0:
            selection = {'triangle': [p['name'] for p in triangle], 'weights': weights.tolist(),
                         'estimate': 'inferred-visible-face-surface'}
            return follow_selection(person, selection), selection
    raise ValueError('点击位置不在清晰的四肢关键点连线或可见面部关键点区域内')


def follow_selection(person, selection):
    if 'triangle' in selection:
        points = {p['name']: p for p in person['keypoints'] if p['confidence'] >= .6}
        vertices = [points.get(name) for name in selection['triangle']]
        if any(p is None for p in vertices):
            raise ValueError('目标面部表面已遮挡')
        return {'pixel': (np.asarray([p['pixel'] for p in vertices]).T@selection['weights']).tolist(),
                'confidence': min(p['confidence'] for p in vertices), 'name': 'face_surface'}
    if 'segment' not in selection:
        return body_pixel(person, selection)[0]
    points = {p['name']: p for p in person['keypoints'] if p['confidence'] >= .6}
    a, b = [points.get(name) for name in selection['segment']]
    if a is None or b is None:
        raise ValueError('目标肢体已遮挡')
    t = selection['fraction']
    delta = np.asarray(b['pixel'])-a['pixel']
    pixel = np.asarray(a['pixel'])+t*delta+selection.get('normalOffset', 0)*np.array([-delta[1], delta[0]])
    return {'pixel': pixel.tolist(),
            'confidence': min(a['confidence'], b['confidence']), 'name': 'limb_surface'}


def surface(frame, keypoint, rig):
    point = metric_keypoint(frame, keypoint)
    if not point['valid'] or point.get('sigmaMeters', 1) > .06:
        raise ValueError(point.get('reason', '双目位置不确定度超过 6 cm'))
    x, y = np.rint(keypoint['pixel']).astype(int)
    depth, valid = frame.depth, frame.valid
    requested = list(keypoint['pixel'])
    if not (0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]):
        raise ValueError('触摸中心像素不在图像内')
    if not valid[y, x]:
        x, y = nearby_surface_pixel(frame, x, y, point['depth'])
        point = metric_keypoint(frame, {**keypoint, 'pixel': [x, y]})
        if not point['valid'] or point['sigmaMeters'] > .06:
            raise ValueError('触摸中心附近没有可靠深度')
    if abs(float(depth[y, x])-point['depth']) > .04:
        raise ValueError('触摸中心位于前景/背景深度边缘')
    # Keep the normal fit local in meters. A fixed pixel patch spans a wrist,
    # clothes and background at close range / wide FOV, mixing unrelated surfaces.
    radius = int(np.clip(round(frame.calibration.fx*.02/point['depth']), 3, 12))
    yy, xx = np.mgrid[max(0, y-radius):min(depth.shape[0], y+radius+1),
                     max(0, x-radius):min(depth.shape[1], x+radius+1)]
    z = depth[yy, xx]
    mask = valid[yy, xx] & (abs(z-point['depth']) < .04)
    c = frame.calibration
    cloud = np.column_stack(((xx[mask]-c.cx)*z[mask]/c.fx, (yy[mask]-c.cy)*z[mask]/c.fy, z[mask]))
    if len(cloud) < 12:
        raise ValueError('触摸表面有效深度不足')
    _, values, axes = np.linalg.svd(cloud-cloud.mean(axis=0), full_matrices=False)
    if values[1] < .001 or values[2]/values[1] > .5:
        raise ValueError('触摸表面法线不稳定')
    normal = axes[-1]
    if normal@np.asarray(point['opticalPosition']) > 0:
        normal = -normal
    position = rig[:3, :3]@point['opticalPosition']+rig[:3, 3]
    return {**point, 'position': position.tolist(), 'normal': (rig[:3, :3]@normal).tolist(),
            'requestedPixel': requested,
            'tangent': (rig[:3, :3]@axes[0]).tolist(), 'frame': 'control-world-y-up',
            'localSurfaceRadius': min(.04, float(values[1]/np.sqrt(len(cloud))))}


def nearby_surface_pixel(frame, x, y, expected):
    yy, xx = np.mgrid[max(0, y-2):min(frame.depth.shape[0], y+3), max(0, x-2):min(frame.depth.shape[1], x+3)]
    keep = frame.valid[yy, xx] & (abs(frame.depth[yy, xx]-expected) < .02)
    distance = (xx-x)**2+(yy-y)**2
    keep &= distance <= 4
    if not np.any(keep):
        raise ValueError('触摸中心 2 像素内没有同表面的有效双目深度')
    candidates = np.column_stack((distance[keep], xx[keep], yy[keep]))
    _, nx, ny = candidates[np.argmin(candidates[:, 0])]
    return int(nx), int(ny)


def check_person_depth(person, target):
    anchors = [point['depth'] for point in person['keypoints'] if point.get('valid')
               and point['name'].endswith(('shoulder', 'hip', 'elbow', 'wrist'))]
    if len(anchors) < 2 or min(abs(np.asarray(anchors)-target['depth'])) > .25:
        raise ValueError('触摸深度无法与同一人物的身体深度关联')


class BodyTargets:
    def __init__(self):
        self.people = []
        self.record = None
        self.targets = {}
        self.records = OrderedDict()

    def update(self, people, frame, pose, timestamp, frame_id, rig):
        if self.record and timestamp <= self.record[2]:
            return people
        used = set()
        for person in people:
            center = person.get('opticalPosition')
            # Identity association uses visual localization, never renderer
            # world translation. Rig calibration remains for touch execution.
            transform = pose
            world = None if center is None or transform is None else transform[:3, :3]@center+transform[:3, 3]
            candidates = []
            if world is not None and self.record and timestamp-self.record[2] < 2.5:
                candidates = [(np.linalg.norm(world-np.asarray(p['_center'])), p['id']) for p in self.people
                              if p.get('_center') is not None and p['id'] not in used]
            candidates.sort()
            unique = candidates and candidates[0][0] < .35 and (len(candidates) == 1 or candidates[1][0]-candidates[0][0] > .12)
            identity = candidates[0][1] if unique else uuid.uuid4().hex[:12]
            person.update(id=identity, frameId=frame_id, _center=None if world is None else world.tolist())
            used.add(identity)
        self.people = copy.deepcopy(people)
        self.record = frame, pose, timestamp, frame_id, rig
        self.records[frame_id] = (self.record, copy.deepcopy(people))
        while len(self.records) > 4:
            self.records.popitem(last=False)
        self.targets = {k: v for k, v in self.targets.items() if time.time()-v['created'] < 60}
        return [{k: v for k, v in person.items() if k != '_center'} for person in people]

    def resolve(self, body):
        if self.record is None or not 0 <= time.time()-self.record[2] < 1.5:
            raise ValueError('人体双目观测过期，请开启第一人称人体检测')
        record, people = self.record, self.people
        if 'pixel' in body and not body.get('targetId'):
            record, people = self.records.get(body.get('frameId'), (None, []))
            if record is None or not 0 <= time.time()-record[2] < 1.5:
                raise ValueError('点击画面已过期，请使用新的检测帧')
        frame, pose, timestamp, frame_id, rig = record
        if rig is None:
            raise ValueError('缺少相机到自身控制坐标的标定')
        token = body.get('targetId')
        selection = self.targets.get(token) if token else None
        if token and selection is None:
            raise ValueError('触摸目标已过期，请重新选择')
        person_id = selection['personId'] if selection else body.get('personId')
        person = next((p for p in people if p['id'] == person_id), None)
        if person is None:
            raise ValueError('目标人物丢失或关联存在歧义，请重新选择')
        if selection:
            point = follow_selection(person, selection)
        else:
            if 'pixel' in body and body.get('frameId') != frame_id:
                raise ValueError('点击画面与检测帧不一致，请重新选择')
            point, selection = body_pixel(person, body)
            token = uuid.uuid4().hex
            selection.update(personId=person_id, created=time.time())
        result = surface(frame, point, rig)
        check_person_depth(person, result)
        self.targets[token] = selection
        return {**result, 'targetId': token, 'personId': person_id, 'selection': selection,
                'observedAt': timestamp, 'frameId': frame_id, 'source': SOURCE,
                'anatomy': {'estimate': selection.get('estimate', 'inferred-limb-surface'),
                            'part': selection.get('bodyPart', selection.get('bodyRegion', point['name'])),
                            'graspVerified': False, 'surfaceSupport': 'local-stereo-patch'}}

    def snapshot(self, include_image=False):
        if self.record is None:
            return {'ready': False, 'people': []}
        frame, _, timestamp, frame_id, _ = self.record
        result = {'ready': True, 'stale': time.time()-timestamp > 1.5, 'frameId': frame_id,
                  'observedAt': timestamp, 'people': [{k: v for k, v in p.items() if k != '_center'} for p in self.people]}
        if include_image:
            ok, image = cv2.imencode('.jpg', frame.rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise RuntimeError('无法编码视觉观测图像')
            result['image'] = 'data:image/jpeg;base64,'+base64.b64encode(image).decode()
        return result
