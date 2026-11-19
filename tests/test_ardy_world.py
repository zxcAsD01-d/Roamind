import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from ardy_keyframes import HIPS, compile_goal, goal_spec
from ardy_playback import WorldPlayback
from ardy_world import FusedWorld
from ardy_to_frontend import PARENTS, VRM_BONES
from realtime import GenerationCancelled


def cloud(walls=(), gap=False, revision=1, map_id='test'):
    floor = [[float(x), 0, float(z)] for x in np.arange(-2.4, 2.4, .08)
             for z in np.arange(-2.4, 2.4, .08) if not gap or abs(x) > .3]
    return FusedWorld({'frame': 'visual-odometry', 'mapId': map_id, 'revision': revision,
                       'floorNormal': [0, 1, 0], 'floorOffset': 0, 'voxelSize': .08,
                       'measuredPoints': floor+list(walls)})


def mock_window(plan, start, history, world, cancelled):
    stop = min(start+6, plan['frames'])
    slots = range(start, stop)
    return {'frames': stop-start, 'fps': 20, 'up': 'yup', 'bones': VRM_BONES, 'parents': PARENTS,
            'rootOrigin': [0, .92, 0], 'footContacts': [0]*((stop-start)*4),
            'localRot': [v for i in slots for v in plan['staticPoses'].get(i, [0, plan['headingTrack'][i], 0]+[0]*78)],
            'rootDelta': (np.asarray(plan['rootTrack'][start:stop])-[0, .92, 0]).ravel().tolist()}


def create(api, action='navigate', target=None):
    return api.create({'perceptionClientId': 'client', 'mapId': 'test', 'start': [0, HIPS, 0],
                       'goal': {'action': action, 'target': target if target is not None else [0, 0, 1.5]}})


def request(api, state, kind, generate=mock_window, **body):
    return api.dispatch(kind, {'sessionId': state['sessionId'], 'revision': state['revision'], **body}, generate)


class ArdyWorldTests(unittest.TestCase):
    def test_walks_around_occupied_wall_and_rejects_unknown_floor(self):
        wall = [[0, float(y), float(z)] for z in np.arange(-.6, .65, .08) for y in np.arange(.12, 1.9, .12)]
        world = cloud(wall)
        start, target = np.array([-1., HIPS, 0.]), [1., 0., 0.]
        plan = compile_goal(world, start, 0, goal_spec({'action': 'navigate', 'target': target}))
        assert len(plan['path']) > 2
        assert abs(plan['path'][-1][0]-1) < .001
        route = np.asarray([start.tolist()]+plan['rootTrack'])[:, [0, 2]]
        assert all(world.clear_route(a, b) for a, b in zip(route, route[1:]))
        with self.assertRaisesRegex(ValueError, '实测地面'):
            compile_goal(cloud(gap=True), start, 0, goal_spec({'action': 'navigate', 'target': target}))


    def test_gesture_poses_are_static_and_world_grounded(self):
        world = cloud()
        plan = compile_goal(world, np.array([0, HIPS, 0]), 0, goal_spec({'action': 'wave', 'target': [0, 1.4, 1.25]}))
        keys = plan['keyframes']
        assert [k['pose'] for k in keys] == ['stand', 'wave-out', 'wave-in', 'wave-out', 'stand']
        assert keys[1]['localRot'] != keys[2]['localRot']
        assert len({k['frame'] for k in keys}) == len(keys)
        for key in keys:
            assert len(key['localRot']) == 81
            assert np.allclose(key['root'], [0, HIPS, 0])
            assert plan['rootTrack'][key['frame']] == key['root']


    def test_replans_at_partial_playback_and_preserves_only_played_history(self):
        api = WorldPlayback(lambda _: cloud())
        state = create(api, 'face', [0, 1.4, 1.25])
        for _ in range(2):
            window = request(api, state, 'next')['window']
            state = request(api, state, 'played', windowId=window['windowId'], throughFrame=window['endFrame'])
        assert state['completed'] and state['playedKeyframes'][0]['frame'] == 11
        state = request(api, state, 'replan', goal={'action': 'navigate', 'target': [0, 0, 1.5]})
        window = request(api, state, 'next')['window']
        assert request(api, state, 'next')['window'] == window
        state = request(api, state, 'played', windowId=window['windowId'], throughFrame=14)
        assert state['playedThrough'] == 14  # 15–17 were generated, not executed
        changed = request(api, state, 'replan', goal={'action': 'face', 'target': [1., 1.3, 0.]})
        assert changed['playedKeyframes'] == state['playedKeyframes']
        assert changed['root'] == state['root']
        assert all(k['frame'] > 14 for k in changed['futureKeyframes'])
        with self.assertRaisesRegex(ValueError, '版本'):
            request(api, state, 'played', windowId=window['windowId'], throughFrame=15)
        with self.assertRaisesRegex(ValueError, '窗口'):
            request(api, changed, 'played', windowId=window['windowId'], throughFrame=15)

        def continued(plan, start, history, world, cancelled):
            assert history['frames'] == 15
            assert history['localRot'][-81:] == window['motion']['localRot'][2*81:3*81]
            assert np.allclose(plan['start'], state['root'])
            return mock_window(plan, start, history, world, cancelled)

        assert request(api, changed, 'next', continued)['window']['startFrame'] == 15


    def test_superseded_inference_does_not_publish_a_window(self):
        api = WorldPlayback(lambda _: cloud())
        state = create(api)
        started, updated = threading.Event(), threading.Event()

        def blocked(plan, start, history, world, cancelled):
            started.set()
            assert updated.wait(5) and cancelled()
            raise GenerationCancelled()

        with ThreadPoolExecutor(1) as pool:
            job = pool.submit(request, api, state, 'next', blocked)
            assert started.wait(5)
            try:
                changed = request(api, state, 'replan', goal={'action': 'face', 'target': [-1, 1, 0]})
            finally:
                updated.set()
            with self.assertRaisesRegex(ValueError, '旧窗口未发布'):
                job.result(timeout=5)
        assert request(api, changed, 'next')['window']['startFrame'] == 0


    def test_failed_replan_or_ack_leaves_current_window_intact(self):
        worlds = [cloud()]
        api = WorldPlayback(lambda _: worlds[-1])
        state = create(api)
        assert goal_spec(state['goal']) == state['goal']
        window = request(api, state, 'next')['window']
        with self.assertRaisesRegex(ValueError, '不得确认'):
            request(api, state, 'played', windowId=window['windowId'], throughFrame=window['endFrame']+1)
        with self.assertRaisesRegex(ValueError, '实测地面'):
            request(api, state, 'replan', goal={'action': 'navigate', 'target': [0, 0, 5]})
        worlds.append(cloud(map_id='reset'))
        with self.assertRaisesRegex(ValueError, '重置'):
            request(api, state, 'replan')
        assert request(api, state, 'next')['window'] == window
        assert request(api, state, 'state')['playedThrough'] == -1
        assert request(api, state, 'close') == {'closed': True}


if __name__ == "__main__":
    unittest.main()
