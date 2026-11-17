"""Bounded image-feature tracking of an annotated region; no scene metadata."""
import cv2
import numpy as np


def track_text(previous, current, match):
    return track_region(previous, current, match)


def track_region(previous, current, match):
    x0, y0, x1, y1 = np.rint(match['box']).astype(int)
    mask = np.zeros_like(previous)
    mask[max(0, y0):y1, max(0, x0):x1] = 255
    points = cv2.goodFeaturesToTrack(previous, 80, .02, 3, mask=mask)
    if points is None or len(points) < 8:
        raise ValueError('标注区域没有足够视觉跟踪特征')
    forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, maxLevel=2)
    if forward is None:
        raise ValueError('标注图像跟踪失败')
    backward, reverse, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, maxLevel=2)
    if backward is None:
        raise ValueError('标注图像反向跟踪失败')
    keep = status.ravel().astype(bool) & reverse.ravel().astype(bool)
    keep &= np.linalg.norm(points-backward, axis=2).ravel() < 1.
    if keep.sum() < 8 or keep.mean() < .7:
        raise ValueError('标注区域遮挡或特征关联不稳定')
    transform, inliers = cv2.estimateAffinePartial2D(points[keep], forward[keep], method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if transform is None or inliers.mean() < .8 or not .85 < np.linalg.norm(transform[:, 0]) < 1.18:
        raise ValueError('标注区域变形过大，请重新选择')
    return tracked_box(previous, current, match, transform)


def tracked_box(previous, current, match, transform):
    x0, y0, x1, y1 = match['box']
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    corners = cv2.transform(corners[None], transform)[0]
    low, high = corners.min(axis=0), corners.max(axis=0)
    height, width = current.shape
    if np.any(low < 0) or high[0] >= width or high[1] >= height:
        raise ValueError('标注区域移出视野')
    x0, y0, x1, y1 = np.rint([*low, *high]).astype(int)
    warped = cv2.warpAffine(previous, transform, (width, height))
    a, b = warped[y0:y1, x0:x1], current[y0:y1, x0:x1]
    similarity = float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])
    if not np.isfinite(similarity) or similarity < .85:
        raise ValueError('标注区域外观改变或被遮挡')
    result = {**match, 'box': [*low.tolist(), *high.tolist()], 'center': ((low+high)/2).tolist()}
    if 'pixel' in match:
        pixel = cv2.transform(np.asarray([[match['pixel']]], np.float32), transform)[0, 0]
        if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
            raise ValueError('标注点移出视野')
        result['pixel'] = pixel.tolist()
    return result
