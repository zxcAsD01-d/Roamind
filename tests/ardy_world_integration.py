"""Real dual-worker ARDY smoke test on a deterministic fused-map fixture."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from ardy_keyframes import HIPS
from ardy_playback import WorldPlayback
from ardy_world import FusedWorld
from dual import DualRunner
from realtime import RealtimeRunner


class WorldArdyIntegration(unittest.TestCase):
    def test_partial_replan_and_static_keys_in_model_output(self):
        floor = [[float(x), 0, float(z)] for x in np.arange(-2, 2, .08) for z in np.arange(-2, 2, .08)]
        world = FusedWorld({'frame': 'visual-odometry', 'mapId': 'smoke', 'revision': 1,
                            'floorNormal': [0, 1, 0], 'floorOffset': 0, 'voxelSize': .08, 'measuredPoints': floor})
        api = WorldPlayback(lambda _: world)
        runner = DualRunner(RealtimeRunner)
        state = api.create({'perceptionClientId': 'smoke', 'mapId': world.id, 'start': [0, HIPS, 0],
                            'goal': {'action': 'navigate', 'target': [0, 0, 1.]}})

        def call(operation, **fields):
            return api.dispatch(operation, {'sessionId': state['sessionId'], 'revision': state['revision'], **fields},
                                runner.world_window)

        try:
            old = call('next')['window']
            state = call('played', windowId=old['windowId'], throughFrame=4)
            root = state['root']
            state = call('replan', goal={'action': 'point', 'target': [0, 1.4, root[2]+1.25]})
            self.assertEqual(state['root'], root)
            self.assertEqual(state['playedThrough'], 4)
            expected = {key['frame']: key for key in state['futureKeyframes']}
            previous = Rotation.from_rotvec(np.asarray(old['motion']['localRot']).reshape(-1, 27, 3)[4])
            max_step = 0.
            while not state['completed']:
                window = call('next')['window']
                self.assertEqual(window['startFrame'], state['playedThrough']+1)
                motion = window['motion']
                for index, local in enumerate(np.asarray(motion['localRot']).reshape(-1, 27, 3)):
                    frame = window['startFrame']+index
                    rotation = Rotation.from_rotvec(local)
                    max_step = max(max_step, float((previous.inv()*rotation).magnitude().max()))
                    previous = rotation
                    if frame in expected:
                        np.testing.assert_allclose(local.ravel(), expected[frame]['localRot'], atol=2e-6)
                        np.testing.assert_allclose(np.array(motion['rootDelta'][index*3:index*3+3])+[0, .92, 0],
                                                   expected[frame]['root'], atol=2e-6)
                state = call('played', windowId=window['windowId'], throughFrame=window['endFrame'])
            self.assertEqual([key['pose'] for key in state['playedKeyframes']], ['stand', 'point', 'point-hold', 'stand'])
            self.assertLess(max_step, .65, 'Successive Core27 rotations must not snap at the handoff or keyframes')
        finally:
            runner.executor.shutdown()
            runner.secondary.close()


if __name__ == '__main__':
    unittest.main()
