"""Isolated CPU MuJoCo feedback loop. This is G1 physics, not VRM physics."""
from __future__ import annotations

import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).resolve().parents[1] / '.local-data/supermimic/g1'
MOTORS = {
    'hip_pitch': (88, .010177520), 'hip_roll': (139, .025101925),
    'hip_yaw': (88, .010177520), 'knee': (139, .025101925),
    'ankle_pitch': (50, .007219450), 'ankle_roll': (50, .007219450),
    'waist_yaw': (88, .010177520), 'waist_roll': (50, .007219450), 'waist_pitch': (50, .007219450),
    'shoulder_pitch': (25, .003609725), 'shoulder_roll': (25, .003609725),
    'shoulder_yaw': (25, .003609725), 'elbow': (25, .003609725),
    'wrist_roll': (25, .003609725), 'wrist_pitch': (5, .00425), 'wrist_yaw': (5, .00425),
}


def motor(name: str) -> tuple[float, float, float, float]:
    key = re.sub(r'^(left_|right_)|_joint$', '', name)
    effort, armature = MOTORS[key]
    frequency = 10 * 2 * np.pi
    return effort, armature, armature * frequency**2, 4 * armature * frequency


def default_positions(names: list[str]) -> np.ndarray:
    defaults = {'hip_pitch': -.312, 'knee': .669, 'ankle_pitch': -.363,
                'elbow': .6, 'shoulder_pitch': .2}
    result = []
    for name in names:
        key = re.sub(r'^(left_|right_)|_joint$', '', name)
        result.append((-.2 if name.startswith('right_') else .2) if key == 'shoulder_roll' else defaults.get(key, 0))
    return np.array(result)


def converted_model(directory: Path) -> ET.Element:
    robot = ET.fromstring((directory / 'g1_29dof_rev_1_0.urdf').read_text())
    compiler = robot.find('mujoco/compiler')
    if compiler is None:
        raise ValueError('G1 URDF lacks MuJoCo compiler settings')
    compiler.attrib.update(meshdir=str(directory), balanceinertia='true', discardvisual='false',
                           fusestatic='false', strippath='false')
    with tempfile.TemporaryDirectory(prefix='supermimic-') as temporary:
        source = Path(temporary) / 'g1.urdf'
        source.write_text(ET.tostring(robot, encoding='unicode'))
        model = mujoco.MjModel.from_xml_path(str(source))
        xml_path = Path(temporary) / 'g1.xml'
        mujoco.mj_saveLastXML(str(xml_path), model)
        return ET.parse(xml_path).getroot()


def build_model(names: list[str], directory: Path = ASSETS) -> mujoco.MjModel:
    xml = converted_model(directory)
    compiler = xml.find('compiler')
    compiler.set('meshdir', str(directory))
    ET.SubElement(xml, 'option', timestep='.005', integrator='implicitfast', gravity='0 0 -9.81', cone='pyramidal')
    world = xml.find('worldbody')
    pelvis = world.find("body[@name='pelvis']")
    if pelvis is None:
        raise ValueError('G1 pelvis was lost during URDF conversion')
    ET.SubElement(pelvis, 'freejoint', name='floating_base_joint')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='100 100 .1',
                  friction='1 .005 .0001', contype='2', conaffinity='1')
    for geom in pelvis.iter('geom'):
        if geom.get('contype') == '0' or geom.get('class') == 'visual':
            continue
        geom.attrib.update(contype='1', conaffinity='2', friction='1 .005 .0001')
    actuators = ET.SubElement(xml, 'actuator')
    for name in names:
        effort, armature, kp, kd = motor(name)
        joint = next(j for j in pelvis.iter('joint') if j.get('name') == name)
        joint.set('armature', str(armature))
        ET.SubElement(actuators, 'position', name=name, joint=name, kp=str(kp), kv=str(kd),
                      forcerange=f'{-effort} {effort}')
    return mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))


class G1Physics:
    def __init__(self, names: list[str], directory: Path = ASSETS):
        self.model = build_model(names, directory)
        self.data = mujoco.MjData(self.model)
        self.q_indices = np.array([self.model.joint(name).qposadr[0] for name in names])
        self.v_indices = np.array([self.model.joint(name).dofadr[0] for name in names])
        self.default = default_positions(names)
        self.scale = np.array([.25 * motor(name)[0] / motor(name)[2] for name in names])
        self.limits = np.array([self.model.joint(name).range for name in names])

    def reset(self, p: np.ndarray, r: np.ndarray, q: np.ndarray, v: np.ndarray,
              w: np.ndarray, qd: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = p
        self.data.qpos[3:7] = np.roll(Rotation.from_matrix(r).as_quat(), 1)
        self.data.qpos[self.q_indices] = q
        self.data.qvel[:3] = v
        self.data.qvel[3:6] = r.T @ w
        self.data.qvel[self.v_indices] = qd
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)

    def state(self) -> tuple[np.ndarray, ...]:
        r = Rotation.from_quat(np.roll(self.data.qpos[3:7], -1)).as_matrix()
        return (self.data.qpos[:3].copy(), r, self.data.qvel[:3].copy(), r @ self.data.qvel[3:6],
                self.data.qpos[self.q_indices].copy(), self.data.qvel[self.v_indices].copy())

    def step(self, action: np.ndarray) -> None:
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError('Expected 29 finite G1 joint actions')
        self.data.ctrl[:] = self.default + self.scale * action
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise ValueError('G1 physics became nonfinite; retain original motion')
