"""Author sparse, static Core27 poses before compiling ARDY root conditions."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from ardy_world import FusedWorld, point3
from core import ARDY_NEUTRAL, ARDY_PARENTS
from touch_ik import forward, solve_frame

FPS = 20
MAX_FRAMES = 3600
HIPS = float(-ARDY_NEUTRAL[:, 1].min())
PROMPTS = {
    'stand': 'A person walks forward.',  # no idle embedding; the stance poses constrain it
    'walk': 'A person walks forward.',
    'wave': 'A person waves their hand.',
    'point': 'A person points forward with their right arm.',
}


def angle_delta(target, current):
    return (target-current+math.pi) % (2*math.pi)-math.pi


def bearing(delta):
    return math.atan2(delta[0], delta[-1])


def facing(local):
    front = Rotation.from_rotvec(np.asarray(local[:3])).apply([0, 0, 1])
    return bearing(front)


def stance(heading, pose, elevation=0.):
    local = np.zeros((27, 3))
    local[8, 2], local[14, 2] = 1.36, -1.36
    local[9, 1], local[15, 1] = .18, -.18
    if pose == 'point':
        wrist = ARDY_NEUTRAL[8]+[.06, .47*math.sin(elevation), .47*math.cos(elevation)]
    elif pose in ('wave-out', 'wave-in'):
        wrist = [-.43 if pose == 'wave-out' else -.25, .76, .18]
    else:
        wrist = None
    if wrist is not None:
        solve_frame(local.ravel(), np.zeros(3), np.asarray(wrist), ARDY_NEUTRAL, ARDY_PARENTS, 10, None)
    local[0, 1] = heading
    return local.ravel().tolist()


def goal_spec(goal):
    if not isinstance(goal, dict) or goal.get('action') not in ('navigate', 'face', 'wave', 'point'):
        raise ValueError('goal.action 必须是 navigate/face/wave/point')
    if set(goal)-{'action', 'target', 'standOff'}:
        raise ValueError('goal 仅接受 action、target 和可选 standOff')
    target = point3(goal.get('target'), 'goal.target')
    distance = goal.get('standOff', 1.25)
    if type(distance) not in (float, int) or not .8 <= distance <= 2.5:
        raise ValueError('standOff 必须在 0.8–2.5 米之间')
    if 'standOff' in goal and goal['action'] not in ('wave', 'point'):
        raise ValueError('只有 wave/point 使用 standOff')
    result = {'action': goal['action'], 'target': target.tolist()}
    if goal['action'] in ('wave', 'point'):
        result['standOff'] = float(distance)
    return result


def gesture_keys(action, root, target, heading):
    if action == 'point':
        shoulder = root[1]+ARDY_NEUTRAL[8, 1]
        elevation = math.atan2(target[1]-shoulder, np.linalg.norm((target-root)[[0, 2]]))
        if abs(elevation) > .65:
            raise ValueError('从当前站位无法自然指向该高度')
        return [(14, 'point', stance(heading, 'point', elevation)),
                (10, 'point-hold', stance(heading, 'point', elevation)), (14, 'stand', stance(heading, 'stand'))]
    if action == 'wave':
        return [(frames, pose, stance(heading, pose)) for frames, pose in
                ((12, 'wave-out'), (9, 'wave-in'), (9, 'wave-out'), (12, 'stand'))]
    return []


def choose_route(world, start, target, action, distance):
    source, destination = start[[0, 2]], target[[0, 2]]
    if np.linalg.norm(destination-source) > 20:
        raise ValueError('单次目标距离不能超过 20 米')
    if action == 'navigate':
        path = world.route(source, destination)
        if path is not None:
            return path
        raise ValueError('目的地没有连续的实测地面或足够净空')
    if action == 'face':
        if np.linalg.norm(destination-source) >= .2:
            return [source]
        raise ValueError('面向目标与角色水平位置过近')
    toward = bearing(source-destination)
    candidates = [source] if abs(np.linalg.norm(source-destination)-distance) <= .18 else []
    candidates.extend(destination+radius*np.array([math.sin(toward+a), math.cos(toward+a)])
                      for radius in (distance, distance+.24)
                      for a in [0, *(sign*k*math.pi/8 for k in range(1, 8) for sign in (1, -1)), math.pi])
    return gesture_route(world, start, target, action, candidates)


def gesture_route(world, start, target, action, candidates):
    for xz in candidates:
        root = np.array([xz[0], world.floor_y(xz)+HIPS, xz[1]])
        if not world.clear_route(xz, xz) or not world.clear_sight(root+[0, .62, 0], target):
            continue
        heading = bearing(target-root)
        try:
            poses = gesture_keys(action, root, target, heading)
        except ValueError:
            continue  # e.g. point from this candidate is outside vertical reach
        joints = [forward(np.asarray(local), root, ARDY_NEUTRAL, ARDY_PARENTS)[0] for _, _, local in poses]
        if not world.clear_body(joints, ARDY_PARENTS):
            continue
        path = world.route(start[[0, 2]], xz)
        if path is not None:
            return path
    raise ValueError('目标附近没有可达、可见且能安全摆出该动作的站位')


class Track:
    def __init__(self, world, start, heading):
        self.world = world
        self.root, self.heading = start, heading
        self.roots, self.headings, self.prompts, self.keys = [], [], [], []

    def segment(self, root, heading, frames, action, pose=None, local=None):
        if len(self.roots)+frames > MAX_FRAMES:
            raise ValueError('ARDY 规划超过 180 秒')
        t = np.arange(1, frames+1)/frames
        ease = (t*t*(3-2*t))[:, None]
        end_heading = self.heading+angle_delta(heading, self.heading)
        self.roots.extend((self.root+(root-self.root)*ease).tolist())
        self.headings.extend((self.heading+(end_heading-self.heading)*ease[:, 0]).tolist())
        self.prompts.extend([PROMPTS[action]]*frames)
        self.root, self.heading = root, end_heading
        if pose is not None:
            local = stance(end_heading, pose) if local is None else local
            joints, _ = forward(np.asarray(local), root, ARDY_NEUTRAL, ARDY_PARENTS)
            if not self.world.clear_body([joints], ARDY_PARENTS):
                raise ValueError(f'静态关键帧 {pose} 身体净空不足')
            self.keys.append({'frame': len(self.roots)-1, 'pose': pose, 'root': root.tolist(),
                              'heading': end_heading, 'localRot': local})

    def turn(self, heading, force=False):
        delta = abs(angle_delta(heading, self.heading))
        if force or delta > .025:
            self.segment(self.root, heading, max(12, math.ceil(1.5*delta/(math.pi/2)*FPS)), 'stand', 'stand')

    def travel(self, xz):
        root = np.array([xz[0], self.world.floor_y(xz)+HIPS, xz[1]])
        distance = np.linalg.norm((root-self.root)[[0, 2]])
        if distance <= .005:
            return
        heading = bearing(root-self.root)
        self.turn(heading)
        self.segment(root, heading, max(12, math.ceil(1.5*distance/.55*FPS)), 'walk', 'stand')

    def document(self, start):
        return {'world': 'fused-pointcloud', 'start': start.tolist(), 'fps': FPS, 'frames': len(self.roots),
                'rootTrack': self.roots, 'headingTrack': self.headings, 'promptTrack': self.prompts,
                'heightTrack': [True]*len(self.roots), 'staticPoses': {k['frame']: k['localRot'] for k in self.keys},
                'keyframes': self.keys}


def compile_goal(world: FusedWorld, start, heading, goal):
    action, target = goal['action'], np.asarray(goal['target'])
    xz = start[[0, 2]]
    floor = world.floor_y(xz)
    if abs(start[1]-floor-HIPS) > .16 or not world.clear_route(xz, xz):
        raise ValueError('起点需要有实测地面、站姿高度及身体净空')
    if action == 'navigate' and abs(target[1]-world.floor_y(target[[0, 2]])) > .18:
        raise ValueError('navigate 的 target 必须接近实测地面高度')
    route = choose_route(world, start, target, action, goal.get('standOff', 1.25))
    track = Track(world, start, heading)
    # Settle a slightly different input hip height without altering the played anchor.
    if abs(start[1]-floor-HIPS) > .001:
        track.segment(np.array([start[0], floor+HIPS, start[2]]), heading, 12, 'stand', 'stand')
    for xz in route[1:]:
        track.travel(xz)
    if action != 'navigate':
        track.turn(bearing(target-track.root), force=True)
        for frames, pose, local in gesture_keys(action, track.root, target, track.heading):
            track.segment(track.root, track.heading, frames, action, pose, local)
    if not track.roots:
        track.segment(track.root, heading, 12, 'stand', 'stand')
    result = track.document(start)
    result['path'] = [[float(x), float(world.floor_y([x, z])), float(z)] for x, z in route]
    return result


def prompt_boundary(plan, start):
    prompt = plan['promptTrack'][start]
    return next((i for i in range(start+1, plan['frames']) if plan['promptTrack'][i] != prompt), plan['frames'])
