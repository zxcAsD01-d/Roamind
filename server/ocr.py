"""On-demand local OCR. Coordinates are pixels, origin top-left, y downward."""
from __future__ import annotations

import base64
import binascii
import math
import threading
import time

from ocr_engine import text_boxes

import cv2
import numpy as np

DIRECTIONS = {'anywhere': (0, 0), 'left': (-1, 0), 'right': (1, 0),
              'up': (0, -1), 'down': (0, 1), 'upper_left': (-1, -1),
              'upper_right': (1, -1), 'lower_left': (-1, 1), 'lower_right': (1, 1)}


def decode(body):
    encoded = body.get('image')
    if not isinstance(encoded, str) or not 1 <= len(encoded) <= 3_000_000:
        raise ValueError('OCR 需要不超过 2 MB 的 base64 JPEG/PNG 图像')
    try:
        data = base64.b64decode(encoded.split(',')[-1], validate=True)
    except binascii.Error as exc:
        raise ValueError('OCR 图像 base64 无效') from exc
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or not 16 <= min(image.shape[:2]) or max(image.shape[:2]) > 1920:
        raise ValueError('OCR 图像需要 JPEG/PNG，边长 16–1920 像素')
    return image


def finite_pair(value, label):
    if not isinstance(value, list) or len(value) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError(label+'需要两个有限数值')
    return value


def select(result, body):
    width, height = result['width'], result['height']
    direction = body.get('direction', 'anywhere')
    if direction not in DIRECTIONS:
        raise ValueError('direction 需要 '+', '.join(DIRECTIONS))
    anchor = finite_pair(body.get('anchor', [width/2, height/2]), 'anchor')
    if not 0 <= anchor[0] < width or not 0 <= anchor[1] < height:
        raise ValueError('anchor 超出图像范围')
    confidence = body.get('minConfidence', .4)
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('minConfidence 需要 0–1')
    sx, sy = DIRECTIONS[direction]
    candidates = []
    for item in result['matches' if body.get('text') else 'lines']:
        x0, y0, x1, y1 = item['box']
        center = [(x0+x1)/2, (y0+y1)/2]
        if item['confidence'] < confidence or sx and sx*(center[0]-anchor[0]) <= 0 or sy and sy*(center[1]-anchor[1]) <= 0:
            continue
        candidates.append({**item, 'center': center})
    candidates.sort(key=lambda item: (round(item['box'][1]/10), item['box'][0]))
    occurrence = body.get('occurrence')
    if occurrence is not None and (type(occurrence) is not int or occurrence < 0):
        raise ValueError('occurrence 需要从 0 开始的序号')
    selected = candidates[occurrence] if occurrence is not None and occurrence < len(candidates) else None
    if occurrence is None and len(candidates) == 1:
        selected = candidates[0]
    return {'direction': direction, 'anchor': anchor, 'candidates': candidates, 'selected': selected,
            'status': 'selected' if selected else 'not_found' if not candidates else 'ambiguous' if occurrence is None else 'index_out_of_range',
            'coordinateFrame': 'image-pixels-top-left-y-down'}


class TextRecognizer:
    def __init__(self):
        self.lock = threading.Lock()
        self.engine = None

    def infer(self, body):
        started = time.perf_counter()
        image = decode(body)
        query = body.get('text', '')
        if not isinstance(query, str) or len(query) > 200 or '\x00' in query:
            raise ValueError('text 需要不超过 200 字符的文字')
        # Validate selection before spending inference time.
        select({'width': image.shape[1], 'height': image.shape[0], 'lines': [], 'matches': []}, body)
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError('OCR 正在处理另一张图像，请稍后重试')
        try:
            result = self.recognize(image, query)
        finally:
            self.lock.release()
        return {**result, **select(result, body), 'query': query, 'source': 'rapidocr-onnx-local',
                'inferenceMs': round((time.perf_counter()-started)*1000, 2)}

    def recognize(self, image, query):
        if self.engine is None:
            from rapidocr_onnxruntime import RapidOCR
            self.engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
        predictions, _ = self.engine(image, return_word_box=True, text_score=.35)
        lines, matches = text_boxes(predictions, query)
        height, width = image.shape[:2]
        return {'width': width, 'height': height, 'lines': lines, 'matches': matches}
