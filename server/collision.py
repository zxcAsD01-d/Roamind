"""Skeleton/box collision reports and conditioning targets for ARDY refinement."""
from __future__ import annotations

import numpy as np

from world import get_world


def project_point(position, radius, objects):
    """Return the closest point outside inflated static boxes, or None."""
    corrected = np.asarray(position, dtype=np.float32).copy()
    hit = False
    for obj in objects:
        center, size = np.asarray(obj["center"]), np.asarray(obj["size"])
        low, high = center-size/2-radius, center+size/2+radius
        if not np.all((corrected > low) & (corrected < high)):
            continue
        distances = np.r_[corrected-low, high-corrected]
        face = int(np.argmin(distances))
        axis = face % 3
        corrected[axis] = (low[axis]-.015) if face < 3 else (high[axis]+.015)
        hit = True
    return corrected if hit else None


def body_collisions(joints, parents):
    """Check both bone endpoints and the intervening bone capsules."""
    objects = get_world()["objects"]
    centers = np.asarray([obj["center"] for obj in objects], dtype=np.float32)
    half = np.asarray([obj["size"] for obj in objects], dtype=np.float32)/2
    ids = np.arange(1, len(parents))
    parent_ids = np.asarray(parents, dtype=int)[1:]
    t = np.array([.25, .5, .75, 1], dtype=np.float32)[None, None, :, None]
    samples = (joints[:, parent_ids, None]*(1-t)+joints[:, ids, None]*t).reshape(-1, 3)
    radii = np.broadcast_to(np.where(ids == 6, .10, np.where(np.isin(ids, [10, 16]), .035, .045))[None, :, None], (len(joints),len(ids),4)).reshape(-1)
    low, high = centers[None]-half[None]-radii[:,None,None], centers[None]+half[None]+radii[:,None,None]
    outside = np.maximum(np.abs(samples[:, None]-centers[None])-half[None], 0)
    hit = (outside*outside).sum(axis=2) < (radii[:, None]-.001)**2
    rows, cols = np.nonzero(hit)
    if not len(rows):
        return {}
    distances = np.concatenate([samples[rows]-low[rows,cols], high[rows,cols]-samples[rows]], axis=1)
    faces = np.argmin(distances, axis=1)
    axes = faces % 3
    projected = samples[rows].copy()
    projected[np.arange(len(rows)),axes] = np.where(faces < 3, low[rows,cols,axes]-.015, high[rows,cols,axes]+.015)
    corrections = {}
    for row, target in zip(rows, projected):
        frame, joint = int(row//(len(ids)*4)), int((row//4)%len(ids)+1)
        corrections[(frame,joint)] = (joints[frame,joint]+target-samples[row]).tolist()
    return corrections


class RootHeightCondition:
    def __init__(self, frames, positions):
        self.frames, self.positions = frames, positions

    def update_constraints(self, data_dict, index_dict):
        data_dict["root_y_pos"].append(self.positions[:, 1])
        index_dict["root_y_pos"].append(self.frames)


class JointClearanceCondition:
    def __init__(self, torch, corrections, offset, shift, device):
        self.indices = torch.tensor([[frame+offset, joint] for frame, joint in corrections], dtype=torch.long)
        positions = np.asarray(list(corrections.values()), dtype=np.float32) + shift
        self.positions = torch.tensor(positions, dtype=torch.float32, device=device)

    def update_constraints(self, data_dict, index_dict):
        data_dict["global_joints_positions"].append(self.positions)
        index_dict["global_joints_positions"].append(self.indices)
