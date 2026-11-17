"""Fit interaction nodes from measured, same-frame stereo pixels; never scene geometry."""
import numpy as np

REST_LABELS = ('sofa', 'couch', 'chair', 'bed', 'bench', '沙发', '椅', '床')
GRASP_LABELS = ('remote', '遥控', 'cup', 'bottle', 'book', 'phone', '杯', '瓶', '书', '手机',
                'sandwich', 'bread', 'apple', 'banana', '三明治', '面包', '苹果', '香蕉')


def extract_nodes(frame, pose, rig, association):
    if rig is None or association.get('confidence', 0) < .4:
        return []
    label = association['label'].lower()
    rest = any(key in label for key in REST_LABELS)
    grasp = any(key in label for key in GRASP_LABELS)
    computer = 'keyboard' in label or '键盘' in label
    if not rest and not grasp and not computer:
        return []
    from semantic_map import SemanticMap
    selected = SemanticMap.surface(frame, association['box'])
    if selected is None:
        return []
    rows, cols = selected
    keep = frame.valid[rows, cols]  # Completed/inferred pixels cannot prove physical support.
    rows, cols = rows[keep], cols[keep]
    if len(rows) < 40:
        return []
    z, c = frame.depth[rows, cols], frame.calibration
    optical = np.column_stack(((cols-c.cx)*z/c.fx, (rows-c.cy)*z/c.fy, z))
    control = optical@rig[:3, :3].T+rig[:3, 3]
    transform = pose@np.linalg.inv(rig)
    pixels = np.column_stack((cols, rows))
    if computer:
        return workstation_nodes(control, pixels, transform)
    return support_nodes(control, pixels, transform) if rest else grasp_nodes(control, pixels, transform)


def workstation_nodes(cloud, pixels, transform):
    if len(cloud) < 40:
        return []
    center = np.median(cloud, axis=0)
    _, values, axes = np.linalg.svd(cloud-center, full_matrices=False)
    if abs(axes[-1, 1]) < .96 or values[1] < .1 or np.std(cloud[:, 1]) > .02:
        return []
    axis = axes[0].copy(); axis[1] = 0; axis /= np.linalg.norm(axis)
    cross = np.cross([0, 1, 0], axis)
    coordinates = np.column_stack((cloud@axis, cloud@cross))
    low, high = np.quantile(coordinates, [.05, .95], axis=0)
    size = high-low
    if min(size) < .08 or not .24 <= max(size) <= .75:
        return []
    occupied = len(np.unique(np.floor((coordinates-low)/.035).astype(int), axis=0))
    if occupied*.035**2/(size[0]*size[1]) < .5:
        return []
    return [node('workstation', center, axis, [size[0], 0, size[1]], len(cloud), np.std(cloud[:, 1]),
                 cloud, pixels, transform)]


def node(kind, center, axis, size, samples, sigma, cloud, pixels, transform):
    index = int(np.argmin(np.linalg.norm(cloud-center, axis=1)))
    return {'kind': kind, 'point': (transform[:3, :3]@center+transform[:3, 3]).round(5).tolist(),
            'controlPoint': center.round(5).tolist(), 'axis': (transform[:3, :3]@axis).round(5).tolist(),
            'controlAxis': axis.round(5).tolist(), 'size': np.asarray(size).round(5).tolist(),
            'pixel': pixels[index].tolist(), 'samples': int(samples), 'sigma': round(float(sigma), 5),
            'source': 'measured-stereo-surface', 'frame': 'control-world-y-up'}


def support_nodes(cloud, pixels, transform):
    candidates = []
    # Horizontal bands are subsequently checked for 2-D support and plane normal.
    for height in np.unique(np.round(cloud[:, 1]/.035)*.035):
        mask = np.abs(cloud[:, 1]-height) < .022
        points = cloud[mask]
        if len(points) < 40:
            continue
        center = np.median(points, axis=0)
        _, values, axes = np.linalg.svd(points-center, full_matrices=False)
        if abs(axes[-1, 1]) < .96 or values[1] < .15:
            continue
        axis = axes[0].copy(); axis[1] = 0; axis /= np.linalg.norm(axis)
        cross = np.cross([0, 1, 0], axis)
        coordinates = np.column_stack((points@axis, points@cross))
        low, high = np.quantile(coordinates, [.05, .95], axis=0)
        size = high-low
        if min(size) < .28 or max(size) < .38:
            continue
        # Sparse edges cannot count as an entire filled seat.
        occupied = len(np.unique(np.floor((coordinates-low)/.08).astype(int), axis=0))
        if occupied*.08**2/(size[0]*size[1]) < .55:
            continue
        center = axis*((low[0]+high[0])/2)+cross*((low[1]+high[1])/2)+[0, float(np.median(points[:, 1])), 0]
        sigma = np.std(points[:, 1])
        candidates.append(node('support', center, axis, [size[0], 0, size[1]], len(points), sigma,
                               points, pixels[mask], transform))
    # Prefer wide support; callers still check seat height against the actor's feet.
    return sorted(candidates, key=lambda row: row['size'][0]*row['size'][2], reverse=True)[:2]


def grasp_nodes(cloud, pixels, transform):
    low, high = np.quantile(cloud, [.05, .95], axis=0)
    size = high-low
    if max(size) > .3 or max(size) < .025 or min(size) > .12:
        return []
    center = np.median(cloud, axis=0)
    return [node('grasp', center, np.array([1., 0, 0]), size, len(cloud), .015, cloud, pixels, transform)]
