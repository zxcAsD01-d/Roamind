"""Bounded planar completion for reconstruction; never supplies measured depth."""
import time

import cv2
import numpy as np


def color_regions(rgb):
    height, width = rgb.shape[:2]
    small = cv2.resize(rgb, (width//2, height//2), interpolation=cv2.INTER_AREA)
    smooth = cv2.GaussianBlur(small, (5, 5), 0)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    edges = cv2.Canny(smooth, 20, 50)
    visited = np.zeros(edges.shape, bool)
    step = max(12, min(edges.shape)//8)
    accepted = 0
    for row in range(step//2, edges.shape[0], step):
        for col in range(step//2, edges.shape[1], step):
            if visited[row, col] or edges[row, col]:
                continue
            flood = np.pad(np.uint8((edges > 0) | visited), 1)
            cv2.floodFill(lab, flood, (col, row), (0, 0, 0), (10, 6, 6), (10, 6, 6),
                          4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (2 << 8))
            region = flood[1:-1, 1:-1] == 2
            visited |= region
            if region.sum() < max(200, region.size*.01):
                continue
            mask = cv2.resize(np.uint8(region), (width, height), interpolation=cv2.INTER_NEAREST)
            yield mask, lab[row, col].astype(np.float32)
            accepted += 1
            if accepted >= 16:
                return


def fit_disparity(pixels, disparity, calibration):
    """A 3-D plane is affine in inverse depth / rectified disparity."""
    design = np.column_stack(((pixels[:, 0]-calibration.cx)/calibration.fx,
                              (pixels[:, 1]-calibration.cy)/calibration.fy, np.ones(len(pixels))))
    rng = np.random.default_rng(23)
    best = np.zeros(len(pixels), bool)
    for _ in range(64):
        ids = rng.choice(len(pixels), 3, replace=False)
        if abs(np.linalg.det(design[ids])) < 1e-4:
            continue
        model = np.linalg.solve(design[ids], disparity[ids])
        inliers = np.abs(design@model-disparity) < .65
        if inliers.sum() > best.sum():
            best = inliers
            if best.mean() > .97:
                break
    if best.sum() < 40 or best.mean() < .8:
        return None
    model, _, rank, _ = np.linalg.lstsq(design[best], disparity[best], rcond=None)
    residual = np.abs(design@model-disparity)
    inliers = residual < .65
    if rank < 3 or inliers.mean() < .8 or np.median(residual[inliers]) > .3:
        return None
    support = pixels[inliers]
    span = np.ptp(support, axis=0)
    if min(span) < 20:
        return None
    # Collinear edge samples cannot determine the interior plane reliably.
    covariance = np.linalg.eigvalsh(np.cov(support.T))
    if covariance[0] < 16:
        return None
    return model, support


def supported_plane(frame, mask, color, lab):
    neighborhood = cv2.dilate(mask, np.ones((9, 9), np.uint8)).astype(bool)
    supported = frame.valid if frame.surface_valid is None else frame.surface_valid
    rows, cols = np.nonzero(neighborhood & supported & (frame.edges == 0))
    similar = np.all(np.abs(lab[rows, cols]-color) < [16, 9, 9], axis=1)
    rows, cols = rows[similar], cols[similar]
    if len(rows) < 40:
        return None
    ids = np.linspace(0, len(rows)-1, min(1200, len(rows))).astype(int)
    pixels = np.column_stack((cols[ids], rows[ids])).astype(float)
    c = frame.calibration
    fit = fit_disparity(pixels, c.fx*c.baseline/frame.depth[rows[ids], cols[ids]], c)
    if fit is None:
        return None
    model, support = fit
    hull = cv2.convexHull(support.astype(np.int32))
    if cv2.contourArea(hull) < max(300, mask.sum()*.2):
        return None
    interior = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(interior, hull, 1)
    interior = cv2.erode(interior, np.ones((3, 3), np.uint8))
    return model, interior.astype(bool) & mask.astype(bool)


def checked_completion(frame, right, model, interior):
    c = frame.calibration
    supported = frame.valid if frame.surface_valid is None else frame.surface_valid
    rows, cols = np.nonzero(interior & ~supported)
    disparity = model[0]*(cols-c.cx)/c.fx+model[1]*(rows-c.cy)/c.fy+model[2]
    depth = c.fx*c.baseline/np.maximum(disparity, .01)
    keep = (disparity > .75) & (depth >= .25) & (depth <= 10)
    # Do not overwrite a conflicting observation, even if its texture is weak.
    keep &= ~frame.valid[rows, cols] | (np.abs(frame.depth[rows, cols]-depth) < np.maximum(.08, depth*.04))
    match_x = cols-disparity
    low = np.floor(match_x).astype(int)
    weight = (match_x-low)[:, None]
    other = (right[rows, np.clip(low, 0, c.width-1)]*(1-weight)
             + right[rows, np.clip(low+1, 0, c.width-1)]*weight)
    color_error = np.max(np.abs(frame.rgb[rows, cols].astype(float)-other), axis=1)
    keep &= (match_x >= 1) & (match_x < c.width-1) & (color_error < 18)
    return rows[keep], cols[keep], depth[keep].astype(np.float32)


def complete_surfaces(frame, right):
    started = time.perf_counter()
    supported = frame.valid if frame.surface_valid is None else frame.surface_valid
    reconstruction = np.where(supported, frame.depth, 0).astype(np.float32)
    inferred = np.zeros(frame.depth.shape, bool)
    planes = 0
    if supported.sum() >= 40:
        lab = cv2.cvtColor(cv2.GaussianBlur(frame.rgb, (9, 9), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        for mask, color in color_regions(frame.rgb):
            if np.count_nonzero(mask.astype(bool) & ~supported) < 100:
                continue
            fit = supported_plane(frame, mask, color, lab)
            if fit is None:
                continue
            rows, cols, depth = checked_completion(frame, right, *fit)
            keep = ~inferred[rows, cols]
            rows, cols, depth = rows[keep], cols[keep], depth[keep]
            if len(rows) < 100:
                continue
            reconstruction[rows, cols] = depth
            inferred[rows, cols] = True
            planes += 1
    frame.reconstruction_depth = reconstruction
    frame.completion_info = {'method': 'color-region-supported-plane', 'planes': planes,
                             'inferredPixels': int(inferred.sum()),
                             'inferenceMs': round((time.perf_counter()-started)*1000, 2),
                             'usage': 'reconstruction-only'}
