"""Convert detected text polygons and character boxes to original image pixels."""
from __future__ import annotations

import numpy as np


def merged(text, confidence, polygons):
    points = np.asarray(polygons, dtype=float).reshape(-1, 2)
    low, high = points.min(axis=0), points.max(axis=0)
    return {'text': text, 'confidence': float(confidence), 'box': [*low.tolist(), *high.tolist()]}


def text_boxes(predictions, query):
    lines, matches = [], []
    needle = ''.join(query.casefold().split())
    for item in predictions or []:
        polygon, text, confidence = item[:3]
        lines.append(merged(text, confidence, [polygon]))
        if not needle:
            continue
        if len(item) < 6:
            if ''.join(text.casefold().split()) == needle:
                matches.append(merged(text, confidence, [polygon]))
            continue
        boxes, tokens, scores = item[3:6]
        normalized, indices = '', []
        for index, token in enumerate(tokens):
            value = ''.join(token.casefold().split())
            normalized += value
            indices.extend([index]*len(value))
        start = 0
        while (found := normalized.find(needle, start)) >= 0:
            selected = sorted(set(indices[found:found+len(needle)]))
            matches.append(merged(query, min(scores[index] for index in selected), [boxes[index] for index in selected]))
            start = found+len(needle)
    return lines, matches
