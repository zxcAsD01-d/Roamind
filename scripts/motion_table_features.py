"""Avatar-independent, yaw-invariant full-body boundaries for motion matching."""
import numpy as np
from scipy.spatial.transform import Rotation

from build_locomotion import BONES, FPS, heading

JOINTS = [index for index, name in enumerate(BONES) if name]


def arrays(clip):
    frames = clip['frames']
    if (type(frames) is not int or not 12 <= frames <= 2400 or clip.get('fps') != FPS
            or clip.get('up') != 'yup' or clip.get('bones') != BONES):
        raise ValueError('incompatible_skeleton_or_rate')
    local = np.asarray(clip['localRot'], dtype=float).reshape(frames, 27, 3)
    root = np.asarray(clip['rootDelta'], dtype=float).reshape(frames, 3)
    contacts = np.asarray(clip['footContacts'], dtype=float).reshape(frames, 4)
    if not all(np.isfinite(value).all() for value in (local, root, contacts)):
        raise ValueError('nonfinite')
    if np.any((contacts < 0) | (contacts > 1)):
        raise ValueError('invalid_contacts')
    return local, root, contacts


def boundary(clip, frame):
    local, root, contacts = arrays(clip)
    before, after = max(0, frame-1), min(len(root)-1, frame+1)
    dt = (after-before)/FPS
    angle = heading(local)[frame]
    alignment = Rotation.from_euler('y', -angle)
    pose = Rotation.from_rotvec(local[frame, JOINTS]).as_quat()
    pose[0] = (alignment*Rotation.from_rotvec(local[frame, 0])).as_quat()
    angular = (Rotation.from_rotvec(local[before, JOINTS]).inv()
               *Rotation.from_rotvec(local[after, JOINTS])).as_rotvec()/dt
    velocity = alignment.apply((root[after]-root[before])/dt)
    support = [contacts[frame, :2].max(), contacts[frame, 2:].max()]
    yaw_rate = (heading(local)[after]-heading(local)[before])/dt
    return np.round(np.r_[velocity[[0, 2]], yaw_rate, support, pose.ravel(), angular.ravel()], 6).tolist()


def costs(query, candidates):
    entries, query = np.asarray(candidates), np.asarray(query)
    count = len(JOINTS)
    quats = entries[:, 5:5+count*4].reshape(-1, count, 4)
    current = query[5:5+count*4].reshape(count, 4)
    angles = 2*np.arccos(np.clip(np.abs(np.sum(quats*current, axis=2)), 0, 1))
    pose = np.mean(angles**2, axis=1)
    velocity = np.sum((entries[:, :2]-query[:2])**2, axis=1)
    angular = np.mean((entries[:, 5+count*4:]-query[5+count*4:])**2, axis=1)
    contact = np.mean(np.abs(entries[:, 3:5]-query[3:5]), axis=1)
    yaw = (entries[:, 2]-query[2])**2
    return pose+.4*velocity+.02*angular+.2*contact+.04*yaw, np.sqrt(pose), angles.max(axis=1)


def describe(clip):
    local, root, _ = arrays(clip)
    indices = np.rint(np.linspace(0, len(root)-1, 9)).astype(int)
    angle = heading(local)
    alignment = Rotation.from_euler('y', -angle[0])
    track = alignment.apply(root-root[0])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(track[:, [0, 2]], axis=0), axis=1))]
    return {'entry': boundary(clip, 0), 'exit': boundary(clip, len(root)-1),
            'trajectory': np.round(np.column_stack((track[indices, 0], track[indices, 2],
                           angle[indices]-angle[0], arc[indices])), 6).tolist(),
            'length': round(float(arc[-1]), 6), 'yaw': round(float(angle[-1]-angle[0]), 6)}
