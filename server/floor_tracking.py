"""Remember a measured floor plane in the visual odometry frame."""
import numpy as np


class FloorTracker:
    def __init__(self):
        self.normal = None
        self.offset = None
        self.foot_clearance = None

    def control_prior(self, pose, rig):
        if self.normal is None or rig is None:
            return None
        optical_normal = pose[:3, :3].T@self.normal
        optical_offset = self.offset+self.normal@pose[:3, 3]
        normal = rig[:3, :3]@optical_normal
        return normal, optical_offset-normal@rig[:3, 3]

    def body_prior(self, rig, optical_foot):
        """Self rig calibration constrains candidates, never supplies depth."""
        if self.foot_clearance is None or rig is None or optical_foot is None:
            return None
        foot = np.asarray(optical_foot, dtype=float)
        if foot.shape != (3,) or not np.isfinite(foot).all():
            raise ValueError('脚底标定需要有限三维坐标')
        control = rig[:3, :3]@foot+rig[:3, 3]
        return np.array([0., 1., 0.]), self.foot_clearance-control[1]

    def update(self, ground, pose, rig, optical_foot=None):
        if (ground is None or rig is None or ground['support'] < 80
                or ground['tiltDeg'] > 12 or ground['residual'] > .04):
            return
        if optical_foot is not None and self.foot_clearance is None:
            foot = np.asarray(optical_foot, dtype=float)
            if foot.shape != (3,) or not np.isfinite(foot).all():
                raise ValueError('脚底标定需要有限三维坐标')
            control = rig[:3, :3]@foot+rig[:3, 3]
            clearance = float(ground['normal']@control+ground['offset'])
            if abs(clearance) <= .18:
                self.foot_clearance = clearance
        optical_normal = rig[:3, :3].T@ground['normal']
        optical_offset = ground['offset']+ground['normal']@rig[:3, 3]
        normal = pose[:3, :3]@optical_normal
        offset = optical_offset-normal@pose[:3, 3]
        if self.normal is None:
            self.normal, self.offset = normal, offset
        else:
            self.normal = .98*self.normal+.02*normal
            length = np.linalg.norm(self.normal)
            self.normal /= length
            self.offset = (.98*self.offset+.02*offset)/length
