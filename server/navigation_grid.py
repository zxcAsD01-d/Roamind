"""Ground occupancy projected from measured stereo points into the visual map."""
import numpy as np

from stereo_depth import height_uncertainty, transform_points


def supported_cells(points, resolution=.175):
    if not len(points):
        return []
    keys, counts = np.unique(np.floor(points[:, [0, 2]]/resolution).astype(int), axis=0, return_counts=True)
    return ((keys[counts >= 3]+.5)*resolution).round(4).tolist()


def navigation_grid(frame, pose, rig, ground):
    if ground is None:
        return None
    points, _, _, _ = frame.points()
    control = transform_points(points, rig)
    plane = ground['coefficients']
    heights = control[:, 1]-plane[0]*control[:, 0]-plane[1]*control[:, 2]-plane[2]
    if rig is None:
        heights = -heights
    uncertainty = height_uncertainty(frame, points, plane, rig)
    floor = np.abs(heights) < .055
    obstacle = (heights > .075+3*uncertainty) & (heights < 2.5)
    world = points@pose[:3, :3].T+pose[:3, 3]
    return {'resolution': .175, 'free': supported_cells(world[floor]),
            'occupied': supported_cells(world[obstacle]), 'frame': 'visual-odometry',
            'source': 'measured-stereo-surface-projection'}
