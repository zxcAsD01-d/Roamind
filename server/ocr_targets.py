"""Text targets from OCR pixels and the same stereo frame, never scene metadata."""
from __future__ import annotations

import copy
import time
import uuid
from collections import OrderedDict

import numpy as np

from touch_vision import surface
from ocr_tracking import track_text


class TextTargets:
    def __init__(self, stereo, recognizer):
        import threading
        self.stereo, self.recognizer = stereo, recognizer
        self.lock = threading.Lock()
        self.targets = OrderedDict()

    def observe(self, body):
        if body.get('resolveDepth') and not body.get('text'):
            raise ValueError('文字触碰需要指定非空目标文字')
        client = body.get('clientId')
        frame, timestamp, frame_id, rig = self.stereo.capture_record(client)
        after = body.get('afterObservedAt', 0)
        if type(after) not in (float, int) or not np.isfinite(after) or timestamp <= after:
            raise ValueError('等待转身停稳后的新双目帧')
        import base64
        import cv2
        ok, data = cv2.imencode('.png', frame.rgb)
        if not ok:
            raise RuntimeError('OCR 视野图像编码失败')
        result = self.recognizer.infer({**body, 'image': base64.b64encode(data).decode()})
        result.update(frameId=frame_id, observedAt=timestamp, clientId=client)
        if body.get('resolveDepth') and result['selected']:
            if rig is None:
                raise ValueError('文字触摸缺少相机到控制坐标的标定')
            identity = 'text-'+uuid.uuid4().hex
            target = self.measure(frame, rig, result['selected'], identity, timestamp, frame_id)
            with self.lock:
                self.targets[identity] = {'clientId': client, 'query': body['text'], 'target': target,
                                          'gray': frame.gray, 'match': result['selected'], 'created': time.monotonic()}
                while len(self.targets) > 32:
                    self.targets.popitem(last=False)
            result['target'] = target
        return result

    @staticmethod
    def measure(frame, rig, match, identity, timestamp, frame_id):
        target = text_surface(frame, rig, match)
        if not 0 <= time.time()-timestamp < 1.5:
            raise ValueError('OCR 源双目帧已过期，请重新识别')
        return {**target, 'targetId': identity, 'personId': identity, 'referenceKind': 'text-surface',
                'text': match['text'], 'box': match['box'], 'observedAt': timestamp, 'frameId': frame_id,
                'source': 'rapidocr+stereo-sgbm-lr'}

    def resolve(self, body):
        identity = body.get('targetId')
        with self.lock:
            record = copy.deepcopy(self.targets.get(identity))
        if record is None or time.monotonic()-record['created'] > 60:
            raise ValueError('文字目标已过期，请重新识别')
        if body.get('clientId') != record['clientId']:
            raise ValueError('文字目标不属于当前视觉会话')
        frame, timestamp, frame_id, rig = self.stereo.capture_record(record['clientId'])
        target = record['target']
        if frame_id == target['frameId'] or time.time()-target['observedAt'] < .35:
            if time.time()-target['observedAt'] >= 1.5:
                raise ValueError('文字接触观测过期')
            return target
        if rig is not None and 0 < timestamp-target['observedAt'] < 1.5:
            try:
                match = track_text(record['gray'], frame.gray, record['match'])
                measured = self.measure(frame, rig, match, identity, timestamp, frame_id)
                if np.linalg.norm(np.asarray(measured['position'])-target['position']) > .18 or np.dot(measured['normal'], target['normal']) < .8:
                    raise ValueError('文字视觉跟踪空间关联不稳定')
            except ValueError:
                pass  # Re-recognize below; failed image tracking never publishes a target.
            else:
                return self.store(identity, measured, frame.gray, match)
        return self.reobserve(record, identity)

    def reobserve(self, record, identity):
        target = record['target']
        observed = self.observe({'clientId': record['clientId'], 'text': record['query']})
        # The OCR frame may advance while recognize runs; use only its matching stereo record.
        frame, timestamp, frame_id, rig = self.stereo.capture_record(record['clientId'], observed['frameId'])
        if rig is None:
            raise ValueError('文字接触缺少相机标定')
        candidates = []
        for match in observed['candidates']:
            try:
                measured = self.measure(frame, rig, match, identity, timestamp, frame_id)
            except ValueError:
                continue  # An unreadable depth patch is never a valid association.
            distance = float(np.linalg.norm(np.asarray(measured['position'])-target['position']))
            if distance <= .18 and np.dot(measured['normal'], target['normal']) >= .8:
                candidates.append((distance, measured, match))
        candidates.sort(key=lambda item: item[0])
        if not candidates or len(candidates) > 1 and candidates[1][0]-candidates[0][0] < .06:
            raise ValueError('文字目标丢失、遮挡或重复文字关联不明确')
        return self.store(identity, candidates[0][1], frame.gray, candidates[0][2])

    def store(self, identity, target, gray, match):
        with self.lock:
            record = self.targets.get(identity)
            if record is None:
                raise ValueError('文字目标已失效')
            if target['observedAt'] > record['target']['observedAt']:
                record.update(target=target, gray=gray, match=match)
            return copy.deepcopy(record['target'])


def text_surface(frame, rig, match):
    """Pick a depth-supported point inside the text region, nearest its center."""
    x0, y0, x1, y1 = match['box']
    offsets = sorted(((dx, dy) for dx in (0, -.12, .12, -.24, .24)
                      for dy in (0, -.12, .12, -.24, .24)), key=lambda v: v[0]**2+v[1]**2)
    failures = []
    for dx, dy in offsets:
        pixel = [(x0+x1)/2+dx*(x1-x0), (y0+y1)/2+dy*(y1-y0)]
        try:
            return surface(frame, {'name': 'text_surface', 'pixel': pixel, 'confidence': match['confidence']}, rig)
        except ValueError as exc:
            failures.append(str(exc))
    raise ValueError('文字区域没有可靠的双目接触表面：'+failures[0])
