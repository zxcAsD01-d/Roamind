"""Conservative head-facing evidence from visible facial landmarks, not bones."""
import math

import numpy as np


def frontal_landmarks(points):
    eyes, nose = points[:2], points[2]
    span = float(np.linalg.norm(eyes[0]-eyes[1]))
    if span < 4:
        return None
    eye_axis = (eyes[1]-eyes[0])/span
    relative = nose-eyes.mean(axis=0)
    offset = abs(float(relative@eye_axis))/span
    vertical = abs(float(relative@np.array([-eye_axis[1], eye_axis[0]])))/span
    yaw = math.degrees(math.atan(2*offset))
    return {'facingCamera': offset < .3 and .05 < vertical < 1.2, 'estimatedYawDegrees': round(yaw, 1),
            'meaning': 'head-facing-estimate-not-eye-contact-or-awareness'}


def person_attention(person, faces):
    box = person.get('box')
    if not isinstance(box, list) or len(box) != 4:
        return {'facingCamera': None, 'source': 'no-associated-face'}
    for face in faces:
        points = np.asarray(face.get('landmarks', []), dtype=float)
        if points.shape != (5, 2) or not np.isfinite(points).all() or face.get('confidence', 0) < .7:
            continue
        nose = points[2]
        if box[0] <= nose[0] <= box[2] and box[1] <= nose[1] <= box[3]:
            estimate = frontal_landmarks(points)
            if estimate is not None:
                return {**estimate, 'source': 'detected-face-five-landmarks'}
    # Hair frequently occludes ears. Two confidently visible eyes and a nose
    # are the central facing evidence; optional ears must not invent it.
    keypoints = {item['name']: item for item in person.get('keypoints', []) if item.get('confidence', 0) >= .8}
    names = ['left_eye', 'right_eye', 'nose']
    if all(name in keypoints for name in names):
        points = np.asarray([keypoints[name]['pixel'] for name in names], dtype=float)
        if points.shape == (3, 2) and np.isfinite(points).all():
            estimate = frontal_landmarks(points)
            if estimate is not None:
                return {**estimate, 'source': 'yolo-high-confidence-eyes-nose'}
    return {'facingCamera': None, 'source': 'facial-landmarks-unverified'}
