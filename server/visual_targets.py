"""Any visible annotated pixel becomes a stereo surface target, independent of class."""
import base64
import copy
import threading
import time
import uuid
from collections import OrderedDict

import cv2
import numpy as np

from ocr_tracking import track_region
from touch_vision import surface
from contact_tracking import check_observation


class VisualTargets:
    def __init__(self, stereo):
        self.stereo = stereo
        self.lock = threading.Lock()
        self.targets = OrderedDict()

    def frame(self, client):
        frame, timestamp, frame_id, _ = self.stereo.capture_record(client)
        ok, encoded = cv2.imencode('.jpg', frame.rgb)
        if not ok:
            raise RuntimeError('视觉标注图像编码失败')
        return {'image': 'data:image/jpeg;base64,'+base64.b64encode(encoded).decode(),
                'frameId': frame_id, 'observedAt': timestamp,
                'width': frame.rgb.shape[1], 'height': frame.rgb.shape[0]}

    def clear(self, client):
        with self.lock:
            for identity in [key for key, value in self.targets.items() if value['clientId'] == client]:
                del self.targets[identity]

    def select(self, body):
        pixel = body.get('pixel')
        if (not isinstance(pixel, list) or len(pixel) != 2
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in pixel)):
            raise ValueError('标注需要有限的 pixel=[x,y]')
        if type(body.get('frameId')) is not int:
            raise ValueError('标注需要画面对应的 frameId')
        label = body.get('label', '视觉标注')
        if not isinstance(label, str) or not 1 <= len(label.strip()) <= 120:
            raise ValueError('标注名称需要 1–120 字符')
        frame, timestamp, frame_id, rig = self.stereo.capture_record(body.get('clientId'), body['frameId'])
        identity = 'point-'+uuid.uuid4().hex
        target = self.measure(frame, rig, pixel, identity, label.strip(), timestamp, frame_id)
        x, y = pixel
        height, width = frame.gray.shape
        match = {'pixel': pixel, 'box': [max(0, x-24), max(0, y-24), min(width-1, x+24), min(height-1, y+24)]}
        with self.lock:
            self.targets[identity] = {'clientId': body['clientId'], 'target': target, 'initial': target,
                                      'gray': frame.gray.copy(), 'match': match, 'created': time.monotonic()}
            while len(self.targets) > 32:
                self.targets.popitem(last=False)
        return target

    @staticmethod
    def measure(frame, rig, pixel, identity, label, timestamp, frame_id):
        if rig is None:
            raise ValueError('视觉标注缺少相机到控制坐标的标定')
        if not 0 <= time.time()-timestamp < 1.5:
            raise ValueError('标注源双目帧已过期')
        target = surface(frame, {'pixel': pixel, 'name': 'annotated_surface', 'confidence': 1.}, rig)
        return {**target, 'targetId': identity, 'referenceKind': 'visual-surface', 'label': label,
                'observedAt': timestamp, 'frameId': frame_id, 'source': 'visual-patch+stereo-sgbm-lr'}

    def resolve(self, body):
        identity = body.get('targetId')
        with self.lock:
            record = copy.deepcopy(self.targets.get(identity))
        if record is None or time.monotonic()-record['created'] > 60:
            raise ValueError('视觉标注已过期，请重新选择')
        if body.get('clientId') != record['clientId']:
            raise ValueError('视觉标注不属于当前会话')
        if record.get('error'):
            raise ValueError(record['error'])
        frame, timestamp, frame_id, rig = self.stereo.capture_record(record['clientId'])
        target = record['target']
        if frame_id == target['frameId']:
            if timestamp != target['observedAt'] or not 0 <= time.time()-timestamp < 1.5:
                raise ValueError('视觉标注帧已失效')
            return target
        if not 0 < timestamp-target['observedAt'] < 1.5:
            raise ValueError('标注跟踪间隔过长，请重新选择')
        match = track_region(record['gray'], frame.gray, record['match'])
        measured = self.measure(frame, rig, match['pixel'], identity, target['label'], timestamp, frame_id)
        check_observation(record['initial'], measured)
        with self.lock:
            current = self.targets.get(identity)
            if current is None:
                raise ValueError('视觉标注已失效')
            if measured['observedAt'] > current['target']['observedAt']:
                current.update(target=measured, gray=frame.gray.copy(), match=match)
            return copy.deepcopy(current['target'])

    def update(self, client):
        """Track at capture cadence, including while ARDY is busy generating."""
        with self.lock:
            expired = [key for key, value in self.targets.items() if time.monotonic()-value['created'] > 60]
            for key in expired:
                del self.targets[key]
            identities = [key for key, value in self.targets.items()
                          if value['clientId'] == client and not value.get('error')]
        for identity in identities:
            try:
                self.resolve({'clientId': client, 'targetId': identity})
            except (ValueError, cv2.error) as exc:
                with self.lock:
                    record = self.targets.get(identity)
                    if record is not None:
                        record['error'] = '标注跟踪已丢失，请重新选择：'+str(exc)
