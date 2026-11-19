"""Metric depth from rectified RGB pairs only; invalid matches stay unknown."""
from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float

    @classmethod
    def read(cls, raw, shape):
        if not isinstance(raw, dict):
            raise ValueError('需要双目内参与基线 calibration')
        values = [float(raw.get(key, float('nan'))) for key in ('fx', 'fy', 'cx', 'cy', 'baseline')]
        if not np.isfinite(values).all():
            raise ValueError('双目标定参数必须为有限数')
        width, height = int(raw.get('width', 0)), int(raw.get('height', 0))
        if (height, width) != shape or not 160 <= width <= 960 or not 90 <= height <= 540:
            raise ValueError('双目图像尺寸与标定不符，支持 160–960 × 90–540')
        fx, fy, cx, cy, baseline = values
        if not (width*.15 < fx < width*4 and height*.15 < fy < height*4
                and 0 <= cx < width and 0 <= cy < height and .03 <= baseline <= .3):
            raise ValueError('双目焦距、主点或基线超出有效范围')
        return cls(width, height, *values)

    def matrix(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], np.float64)


@dataclass
class DepthFrame:
    rgb: np.ndarray
    gray: np.ndarray
    depth: np.ndarray
    valid: np.ndarray
    edges: np.ndarray
    calibration: Calibration
    surface_valid: np.ndarray | None = None
    reconstruction_depth: np.ndarray | None = None
    completion_info: dict[str, float | int | str] | None = None

    def points(self, stride=3):
        supported = self.valid if self.surface_valid is None else self.surface_valid
        return self._sample(self.depth, supported, stride)

    def mapping_points(self, stride=4):
        if self.reconstruction_depth is None:
            return self.points(stride)
        return self._sample(self.reconstruction_depth, self.reconstruction_depth > 0, stride)

    def _sample(self, depth, supported, stride):
        rows, cols = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        keep = supported[rows, cols]
        rows, cols = rows[keep], cols[keep]
        z = depth[rows, cols]
        c = self.calibration
        xyz = np.column_stack(((cols-c.cx)*z/c.fx, (rows-c.cy)*z/c.fy, z))
        return xyz, self.rgb[rows, cols, ::-1], rows, cols


def decode_image(encoded):
    if not isinstance(encoded, str) or len(encoded) > 2_500_000:
        raise ValueError('双目图像必须是 JPEG/PNG，编码长度不超过 2.5 MB')
    data = np.frombuffer(base64.b64decode(encoded.split(',')[-1], validate=True), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('无法解码双目图像')
    return image


def epipolar_texture(gray):
    """Horizontal matching needs horizontal texture, not only row edges."""
    gradient = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)/8
    return cv2.boxFilter(gradient*gradient, -1, (7, 7)) > 1.


def coherent_matches(disparity, supported):
    """Require local disparity support without averaging across depth edges."""
    height, width = disparity.shape
    padded = np.pad(disparity, 1)
    valid = np.pad(supported, 1)
    tolerance = np.clip(.03*disparity, .85, 1.5)
    neighbors = np.zeros(disparity.shape, dtype=np.uint8)
    for y, x in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)):
        area = np.s_[y:y+height, x:x+width]
        neighbors += valid[area] & (np.abs(padded[area]-disparity) <= tolerance)
    # A one-pixel-wide feature can still have two consistent neighbors along it.
    return supported & (neighbors >= 2)


def free_space_conflicts(frame, optical, voxel_size):
    """Only dense, directly measured depth behind a projected voxel disproves it."""
    c = frame.calibration
    result = np.zeros(len(optical), dtype=bool)
    front = np.flatnonzero((optical[:, 2] >= .25) & (optical[:, 2] <= 10.))
    x, y, z = optical[front].T
    u, v = np.rint(c.fx*x/z+c.cx), np.rint(c.fy*y/z+c.cy)
    inside = (u >= 2) & (u < c.width-2) & (v >= 2) & (v < c.height-2)
    if not inside.any():
        return result
    measured = frame.valid if frame.surface_valid is None else frame.surface_valid
    depth = np.where(measured, frame.depth, 0)
    nearest = cv2.erode(depth, np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    z = z[inside]
    # Reserve room for the voxel extent and pixel-disparity uncertainty.
    margin = 1.5*voxel_size+z*z/(c.fx*c.baseline)
    result[front[inside]] = nearest[v[inside].astype(int), u[inside].astype(int)] > z+margin
    return result


class StereoDepth:
    def __init__(self):
        self.matchers = {}

    def _matcher(self, disparities, reverse=False):
        key = (disparities, reverse)
        if key not in self.matchers:
            self.matchers[key] = cv2.StereoSGBM_create(
                minDisparity=-disparities if reverse else 0, numDisparities=disparities,
                blockSize=5, P1=8*25, P2=32*25, disp12MaxDiff=1,
                uniquenessRatio=10, speckleWindowSize=60, speckleRange=2,
                preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        return self.matchers[key]

    def infer(self, left, right, calibration):
        if left.shape != right.shape:
            raise ValueError('左右眼必须同尺寸、同步、平行光轴且已校正')
        gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        other = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        disparities = min(96, max(32, (left.shape[1]//5)//16*16))
        forward = self._matcher(disparities).compute(gray, other).astype(np.float32)/16
        backward = self._matcher(disparities, True).compute(other, gray).astype(np.float32)/16
        rows, cols = np.indices(gray.shape)
        match_x = np.rint(cols-forward).astype(np.int32)
        inside = (match_x >= 0) & (match_x < gray.shape[1])
        reverse = backward[rows, np.clip(match_x, 0, gray.shape[1]-1)]
        valid = inside & (forward > .75) & (np.abs(forward+reverse) < 1.25)
        mean = cv2.boxFilter(gray.astype(np.float32), -1, (7, 7))
        variance = cv2.boxFilter(gray.astype(np.float32)**2, -1, (7, 7))-mean**2
        valid &= variance > 3
        depth = calibration.fx*calibration.baseline/np.maximum(forward, .01)
        valid &= (depth >= .25) & (depth <= 10)
        depth[~valid] = 0
        edges = cv2.Canny(gray, 45, 110)
        jumps = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
        edges[(jumps > .2) & valid & np.roll(valid, 1, axis=1)] = 255
        # Optical flow can retain an existing RANSAC-verified landmark through
        # weak texture. New floor/obstacle surfaces need epipolar support too.
        surface_valid = coherent_matches(forward, valid & epipolar_texture(gray))
        frame = DepthFrame(left, gray, depth, valid, edges, calibration, surface_valid)
        from surface_completion import complete_surfaces
        complete_surfaces(frame, right)
        return frame


def transform_points(points, transform=None):
    """将光学坐标变换到控制/重力坐标；没有外参时保留旧坐标。"""
    if transform is None:
        return points.copy()
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('点云变换必须是有限的 4x4 矩阵')
    rotation, translation = matrix[:3, :3], matrix[:3, 3]
    return points@rotation.T+translation


def _plane_coefficients(normal, offset):
    if abs(normal[1]) < 1e-6:
        return None
    return np.array([-normal[0]/normal[1], -normal[2]/normal[1], -offset/normal[1]], np.float64)


def _ground_candidates(control, transform):
    gravity_sign = 1. if transform is not None else -1.
    camera = np.zeros(3) if transform is None else np.asarray(transform, dtype=np.float64)[:3, 3]
    below = -gravity_sign*(control[:, 1]-camera[1])
    distance = np.linalg.norm((control-camera)[:, [0, 2]], axis=1)
    candidates = control[(below > .05) & (below < 4.5) & (distance < 8)]
    return candidates, gravity_sign, camera


def _ransac_ground(candidates, gravity_sign, camera):
    if len(candidates) < 80:
        return None
    camera_y = float(camera[1])
    rng = np.random.default_rng(7)
    best, best_score = None, -1.
    for _ in range(64):
        sample = candidates[rng.choice(len(candidates), 3, replace=False)]
        normal = np.cross(sample[1]-sample[0], sample[2]-sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-6:
            continue
        normal /= length
        if normal[1]*gravity_sign < 0:
            normal = -normal
        if abs(normal[1]) < .92:
            continue
        offset = -float(normal@sample[0])
        coefficients = _plane_coefficients(normal, offset)
        if coefficients is None:
            continue
        floor_y = coefficients[0]*camera[0]+coefficients[1]*camera[2]+coefficients[2]
        floor_below = -gravity_sign*(floor_y-camera_y)
        if not .15 < floor_below < 3.5:
            continue
        residual = np.abs(candidates@normal+offset)
        support = int(np.sum(residual < .045))
        close = residual[residual < .12]
        score = support - 2.*float(np.median(close) if len(close) else .12)
        if support >= 40 and score > best_score:
            best, best_score = (normal, offset), score
    return best


def _finalize_ground(candidates, best, gravity_sign, frame_name):
    if best is None:
        return None
    normal, offset = best
    residual = np.abs(candidates@normal+offset)
    inliers = candidates[residual < .045]
    if len(inliers) < 40:
        return None
    centered = inliers-inliers.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    normal = axes[-1]
    if normal[1]*gravity_sign < 0:
        normal = -normal
    offset = -float(normal@inliers.mean(axis=0))
    coefficients = _plane_coefficients(normal, offset)
    if coefficients is None:
        return None
    tilt = float(np.degrees(np.arccos(np.clip(abs(normal[1]), 0., 1.))))
    return {'coefficients': coefficients, 'normal': normal, 'offset': offset,
            'support': int(len(inliers)), 'residual': float(np.median(np.abs(inliers@normal+offset))),
            'tiltDeg': tilt, 'frame': frame_name}


def ground_estimate(points, transform=None, prior=None):
    """用 RANSAC 找到有足够支持的地面，并以控制坐标的 Y 轴作为重力参考。"""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('点云必须为 N×3 数组')
    control = transform_points(points, transform)
    candidates, gravity_sign, camera = _ground_candidates(control, transform)
    if prior is not None:
        normal, offset = prior
        candidates = candidates[np.abs(candidates@normal+offset) < .12]
    best = _ransac_ground(candidates, gravity_sign, camera)
    frame_name = 'gravity-aligned-y-up' if transform is not None else 'legacy-optical'
    return _finalize_ground(candidates, best, gravity_sign, frame_name)


def ground_plane(points, transform=None):
    estimate = ground_estimate(points, transform)
    return None if estimate is None else estimate['coefficients']


def _unknown_sectors():
    return [{'angle': angle, 'clearance': None, 'confidence': 0.} for angle in range(-45, 46, 15)]


def _horizontal_view(control, transform):
    if transform is None:
        return control[:, 2], control[:, 0]
    matrix = np.asarray(transform, dtype=np.float64)
    forward = matrix[:3, 2].copy()
    forward[1] = 0
    length = np.linalg.norm(forward)
    if length < 1e-6:
        return None
    forward /= length
    right = np.cross(forward, np.array([0., 1., 0.]))
    relative = control-matrix[:3, 3]
    return relative@forward, relative@right


def height_uncertainty(frame, points, plane, transform):
    calibration = getattr(frame, 'calibration', None)
    if calibration is None:
        return np.zeros(len(points))
    normal = np.array([-plane[0], 1., -plane[1]])
    if transform is not None:
        normal = np.asarray(transform)[:3, :3].T@normal
    # Half-pixel disparity uncertainty grows quadratically with depth. Its
    # vertical component must not turn a noisy floor sample into an obstacle.
    return np.abs(points@normal)*points[:, 2]*.5/(calibration.fx*calibration.baseline)


def clearance_details(frame, transform=None, ground_prior=None):
    points, _, _, _ = frame.points()
    estimate = ground_estimate(points, transform, ground_prior)
    if estimate is None:
        return _unknown_sectors(), None, None
    control = transform_points(points, transform)
    gravity_sign = 1. if transform is not None else -1.
    plane = estimate['coefficients']
    floor_y = plane[0]*control[:, 0]+plane[1]*control[:, 2]+plane[2]
    heights = gravity_sign*(control[:, 1]-floor_y)
    floor = np.abs(heights) < .055
    uncertainty = height_uncertainty(frame, points, plane, transform)
    obstacles = (heights > .075+3*uncertainty) & (heights < 2.5)
    view = _horizontal_view(control, transform)
    if view is None:
        return _unknown_sectors(), plane.round(5).tolist(), estimate
    forward, right = view
    sectors = []
    for angle in range(-45, 46, 15):
        radians = np.deg2rad(angle)
        distance = forward*np.cos(radians)+right*np.sin(radians)
        lateral = right*np.cos(radians)-forward*np.sin(radians)
        corridor = (np.abs(lateral) < .40) & (distance > .15)
        free = distance[corridor & floor]
        hits = distance[corridor & obstacles]
        coverage = min(1., len(free)/100)
        reach = float(np.percentile(free, 85))-.2 if len(free) >= 25 else None
        obstacle = float(np.percentile(hits, 3))-.15 if len(hits) >= 5 else None
        clearance = min(reach, obstacle) if reach is not None and obstacle is not None else reach
        sectors.append({'angle': angle, 'clearance': None if clearance is None else round(max(0., clearance), 3),
                        'confidence': round(coverage, 3), 'obstacleDistance': obstacle})
    return sectors, plane.round(5).tolist(), estimate


def clearance_sectors(frame, transform=None):
    sectors, plane, _ = clearance_details(frame, transform)
    return sectors, plane


def depth_preview(frame):
    depth = frame.depth if frame.reconstruction_depth is None else frame.reconstruction_depth
    normalized = np.uint8(np.clip(1-depth/8, 0, 1)*255)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[depth <= 0] = [18, 22, 24]
    supported = frame.valid if frame.surface_valid is None else frame.surface_valid
    inferred = (depth > 0) & ~supported if frame.reconstruction_depth is not None else np.zeros(depth.shape, bool)
    color[inferred] = color[inferred]*.55+np.array([65, 35, 15])
    color[(frame.edges > 0) & frame.valid] = [230, 240, 240]
    ok, encoded = cv2.imencode('.png', cv2.resize(color, (320, 180)))
    if not ok:
        raise RuntimeError('深度预览编码失败')
    return 'data:image/png;base64,'+base64.b64encode(encoded).decode()
