"""Bounded motion-space refinement of physical corrections, preserving source timing."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from supermimic_skeleton import HumanMotion


def smooth_residual(values: np.ndarray, sigma: float) -> np.ndarray:
    if not sigma:
        return values
    filtered = gaussian_filter1d(values, sigma, axis=0)
    # Initial physical state equals the reference: filtering must not move the entry pose.
    filtered[0] = values[0]
    return filtered


def residual_candidates(source: HumanMotion, physical: dict) -> Iterator[tuple[dict, dict]]:
    motion = HumanMotion.load(physical)
    original = Rotation.from_rotvec(source.local.reshape(-1, 3))
    correction = (Rotation.from_rotvec(motion.local.reshape(-1, 3)) * original.inv()).as_rotvec()
    correction = correction.reshape(source.local.shape)
    translation = motion.root - source.root
    for seconds in (0., .04, .08, .12):
        rotations = smooth_residual(correction, seconds * source.document['fps'])
        displacement = smooth_residual(translation, seconds * source.document['fps'])
        for strength in (1., .5, .25, .125):
            local = (Rotation.from_rotvec((rotations * strength).reshape(-1, 3)) * original).as_rotvec()
            root = source.root + displacement * strength
            settings = {'strength': strength, 'smoothingSeconds': seconds,
                        'footMode': physical['supermimicReconstruction']['footMode']}
            document = {**physical, 'localRot': local.ravel().tolist(), 'rootDelta': root.ravel().tolist(),
                        'supermimicRefinement': settings}
            yield document, settings
