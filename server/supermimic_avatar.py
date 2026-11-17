"""Calibrate against measured VRM rest geometry without modifying the retargeter."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic_skeleton import BODY_MAP, HumanMotion


@dataclass
class AvatarCalibration:
    source: HumanMotion
    human: HumanMotion
    source_indices: list[int]
    root_scale: float
    name: str

    @classmethod
    def load(cls, source: HumanMotion, profile: dict, root_scale: float | None = None) -> AvatarCalibration:
        positions = profile.get('positions', {})
        if profile.get('version') != 1 or not all(name in positions for name in BODY_MAP):
            raise ValueError('Avatar profile requires measured humanoid rest positions')
        # Match VrmRetargeter: only the deepest source joint owns a semantic bone.
        indices = sorted(index for name, index in source.semantic.items() if name in positions)
        parents = effective_parents(source.document['parents'], indices)
        rest = np.array([positions[source.document['bones'][i]] for i in indices], dtype=float)
        if rest.shape != (len(indices), 3) or not np.isfinite(rest).all():
            raise ValueError('Avatar rest positions must be finite 3D vectors')
        scale = root_scale if root_scale is not None else profile.get('rootScales', {}).get(source.document['name'])
        if scale is None:
            raise ValueError('Avatar profile lacks measured root scale for this clip; provide --root-scale')
        if type(scale) not in (float, int) or not np.isfinite(scale) or not .2 <= scale <= 3:
            raise ValueError('Avatar root scale must be finite and between 0.2 and 3')
        world = source.world[:, indices]
        local = local_rotations(world, parents)
        document = copy.deepcopy(source.document)
        document.update(bones=[source.document['bones'][i] for i in indices], parents=parents,
                        restJoints=(rest - rest[0]).ravel().tolist(), localRot=local.ravel().tolist(),
                        rootDelta=(source.root * scale).ravel().tolist())
        document.pop('footContacts', None)
        return cls(source, HumanMotion.load(document), indices, float(scale), str(profile.get('avatar', 'measured VRM')))

    def restore(self, corrected: dict) -> dict:
        motion = HumanMotion.load(corrected)
        if motion.document['bones'] != self.human.document['bones'] or len(motion.root) != len(self.source.root):
            raise ValueError('Avatar correction skeleton or frame count changed')
        residual = motion.world @ self.human.world.transpose(0, 1, 3, 2)
        mapping = {index: residual[:, i] for i, index in enumerate(self.source_indices)}
        world = self.source.world.copy()
        for joint, parent in enumerate(self.source.document['parents']):
            if joint in mapping:
                world[:, joint] = mapping[joint] @ self.source.world[:, joint]
            elif parent >= 0:
                world[:, joint] = world[:, parent] @ Rotation.from_rotvec(self.source.local[:, joint]).as_matrix()
        output = copy.deepcopy(self.source.document)
        output.update(localRot=local_rotations(world, self.source.document['parents']).ravel().tolist(),
                      rootDelta=(motion.root / self.root_scale).ravel().tolist(),
                      name=corrected['name'], source=corrected['source'])
        for key in ('footContacts', 'search', 'quality', 'footLift', 'distance', 'angle', 'heightOffset'):
            output.pop(key, None)
        for key in ('supermimicReconstruction', 'supermimicRefinement'):
            if key in corrected:
                output[key] = corrected[key]
        return output


def effective_parents(parents: list[int], indices: list[int]) -> list[int]:
    lookup, result = {joint: i for i, joint in enumerate(indices)}, []
    for joint in indices:
        parent = parents[joint]
        while parent >= 0 and parent not in lookup:
            parent = parents[parent]
        result.append(lookup[parent] if parent >= 0 else -1)
    return result


def local_rotations(world: np.ndarray, parents: list[int]) -> np.ndarray:
    local = world.copy()
    for joint, parent in enumerate(parents):
        if parent >= 0:
            local[:, joint] = world[:, parent].transpose(0, 2, 1) @ world[:, joint]
    return Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(*world.shape[:2], 3)
