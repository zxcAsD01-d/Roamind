"""Receding-horizon execution: generated samples are provisional until played."""
from __future__ import annotations

import json
import math
import os
import threading
import uuid
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from ardy_keyframes import MAX_FRAMES, compile_goal, facing, goal_spec
from ardy_world import FusedWorld, point3
from realtime import GenerationCancelled

HISTORY_LIMIT = 160
MAX_SESSIONS = 16


def read_world(client: str) -> FusedWorld:
    base = os.environ.get('MOTION_PERCEPTION_URL', 'http://127.0.0.1:8200').rstrip('/')
    try:
        query = urlencode({'clientId': client})
        with urlopen(f'{base}/api/stereo/planning-map?{query}', timeout=5) as response:
            return FusedWorld(json.load(response))
    except HTTPError as exc:
        raise ValueError(f'无法取得融合点云（HTTP {exc.code}）：{exc.read(2048).decode(errors="replace")}') from exc
    except (TimeoutError, URLError) as exc:
        raise ValueError('连接感知服务失败，请确认服务及双目会话在线') from exc


def key(value, label):
    if not isinstance(value, str) or not 1 <= len(value) <= 100:
        raise ValueError(f'{label} 无效')
    return value


def append_history(history, motion, begin, end):
    frames = min(HISTORY_LIMIT, (history['frames'] if history else 0)+end-begin)
    result = {name: motion[name] for name in ('fps', 'up', 'bones', 'parents', 'rootOrigin')}
    if history and any(history[name] != value for name, value in result.items()):
        raise ValueError('已播放历史与生成窗口的骨架或根坐标不一致')
    for name, stride in (('localRot', 81), ('rootDelta', 3), ('footContacts', 4)):
        old = history[name] if history else []
        result[name] = (old+motion[name][begin*stride:end*stride])[-frames*stride:]
    return {**result, 'frames': frames}


class Playback:
    def __init__(self, client, world, start, heading, goal):
        self.id = f'ardy_world_{uuid.uuid4().hex}'
        self.client, self.world, self.goal = client, world, goal
        self.revision, self.played, self.offset = 1, -1, 0
        self.root, self.heading = start, heading
        self.plan = self._compile(world, goal, 1, 0, start, heading)
        self.history = self.window = self.last_window = None
        self.generating = self.closed = False
        self.lock = threading.RLock()
        self.passed_keys = []

    def _compile(self, world, goal, revision, offset, root, heading):
        plan = compile_goal(world, root, heading, goal)
        if offset+plan['frames'] > MAX_FRAMES:
            raise ValueError('同一 ARDY 会话累计最多 180 秒，请另开会话')
        return {**plan, 'id': f'{self.id}_r{revision}'}

    def _check(self, revision):
        if self.closed or type(revision) is not int or revision != self.revision:
            raise ValueError('会话已关闭或规划版本已过期，请读取最新状态')

    def _keys(self):
        return [{**item, 'frame': item['frame']+self.offset, 'revision': self.revision}
                for item in self.plan['keyframes'] if item['frame']+self.offset > self.played]

    def state(self):
        with self.lock:
            return {'sessionId': self.id, 'revision': self.revision, 'mapId': self.world.id,
                    'mapRevision': self.world.revision, 'frame': 'visual-odometry', 'goal': self.goal,
                    'playedThrough': self.played, 'root': self.root.tolist(), 'heading': self.heading,
                    'completed': self.played+1 == self.offset+self.plan['frames'],
                    'fps': self.plan['fps'], 'endFrame': self.offset+self.plan['frames']-1,
                    'playedKeyframes': self.passed_keys.copy(), 'futureKeyframes': self._keys(),
                    'path': self.plan['path']}

    def replan(self, revision, goal, load: Callable):
        with self.lock:
            self._check(revision)
            played, root, heading = self.played, self.root, self.heading
            goal = self.goal if goal is None else goal_spec(goal)
            map_id, map_revision = self.world.id, self.world.revision
        world = load(self.client)
        if world.id != map_id or world.revision < map_revision:
            raise ValueError('点云会话已重置或版本落后，不能复用旧世界的播放历史')
        plan = self._compile(world, goal, revision+1, played+1, root, heading)
        with self.lock:
            self._check(revision)
            if self.played != played:
                raise ValueError('规划期间播放游标已推进，请从最新确认帧重试')
            self.revision += 1
            self.offset, self.world, self.goal, self.plan = played+1, world, goal, plan
            self.window = self.last_window = None
            return self.state()

    def next(self, revision, generate: Callable):
        with self.lock:
            self._check(revision)
            if self.window:
                return self.window_response()
            start = self.played+1-self.offset
            if start == self.plan['frames']:
                return self.window_response()
            if self.generating:
                raise ValueError('上一个生成请求正在收尾，请稍后重试')
            self.generating = True
            plan, history, world = self.plan, self.history, self.world
        return self._generate(revision, start, generate, plan, history, world)

    def window_response(self):
        return {'sessionId': self.id, 'revision': self.revision, 'playedThrough': self.played,
                'window': self.window, 'completed': self.played+1 == self.offset+self.plan['frames']}

    def _generate(self, revision, start, generate, plan, history, world):
        def cancelled():
            with self.lock:
                return self.closed or revision != self.revision

        try:
            motion = generate(plan, start, history, world, cancelled)
            with self.lock:
                self._check(revision)
                first = self.offset+start
                self.window = {'windowId': uuid.uuid4().hex, 'startFrame': first,
                               'endFrame': first+motion['frames']-1, 'motion': {**motion, 'startFrame': first}}
                return self.window_response()
        except GenerationCancelled as exc:
            raise ValueError('规划已更新或会话已关闭，旧窗口未发布') from exc
        finally:
            with self.lock:
                self.generating = False

    def acknowledge(self, revision, window_id, through):
        with self.lock:
            self._check(revision)
            if type(through) is not int or through < 0:
                raise ValueError('throughFrame 必须是非负整数')
            if not self.window or window_id != self.window['windowId']:
                last = self.last_window
                if last and window_id == last['windowId'] and last['startFrame'] <= through <= last['endFrame']:
                    return self.state()
                raise ValueError('播放窗口已失效或不属于当前规划')
            window = self.window
            first = window['startFrame']
            if not first <= through <= window['endFrame']:
                raise ValueError('不得确认尚未发布的帧')
            if through > self.played:
                self._commit(window, through)
                if through == window['endFrame']:
                    self.last_window = {name: window[name] for name in ('windowId', 'startFrame', 'endFrame')}
                    self.window = None
            return self.state()

    def _commit(self, window, through):
        first, motion = window['startFrame'], window['motion']
        history = append_history(self.history, motion, self.played+1-first, through+1-first)
        root = np.asarray(history['rootDelta'][-3:])+[0, .92, 0]
        heading = facing(history['localRot'][-81:])
        keys = [item for item in self._keys() if item['frame'] <= through]
        self.history, self.root, self.heading, self.played = history, root, heading, through
        self.passed_keys.extend(keys)


class WorldPlayback:
    def __init__(self, load=read_world):
        self.load = load
        self.lock = threading.Lock()
        self.sessions: dict[str, Playback] = {}

    def create(self, body):
        client = key(body.get('perceptionClientId'), 'perceptionClientId')
        map_id = key(body.get('mapId'), 'mapId')
        start, goal = point3(body.get('start'), 'start'), goal_spec(body.get('goal'))
        heading = body.get('startHeading', 0)
        if type(heading) not in (int, float) or not -360 <= heading <= 360:
            raise ValueError('startHeading 必须是 -360–360 度')
        world = self.load(client)
        if world.id != map_id:
            raise ValueError('mapId 已过期，不能在不同坐标系间复用位置')
        session = Playback(client, world, start, math.radians(heading), goal)
        with self.lock:
            if len(self.sessions) >= MAX_SESSIONS:
                raise ValueError('ARDY 会话已满，请关闭不用的会话')
            self.sessions[session.id] = session
        return session.state()

    def dispatch(self, operation, body, generate):
        if operation == 'create':
            return self.create(body)
        session_id = key(body.get('sessionId'), 'sessionId')
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise ValueError('ARDY 会话不存在或已关闭')
        revision = body.get('revision')
        if operation == 'state':
            return session.state()
        if operation == 'next':
            return session.next(revision, generate)
        if operation == 'played':
            return session.acknowledge(revision, key(body.get('windowId'), 'windowId'), body.get('throughFrame'))
        if operation == 'replan':
            return session.replan(revision, body.get('goal'), self.load)
        if operation == 'close':
            with self.lock, session.lock:
                session._check(revision)
                session.closed = True
                del self.sessions[session_id]
            return {'closed': True}
        raise ValueError('未知的 ARDY 会话操作')
