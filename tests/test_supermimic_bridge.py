"""Length calibration, coordinate roundtrip, source preservation and physical quality gates."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from supermimic_bridge import adapt_human, reverse_human, track_adapted
from supermimic_avatar import AvatarCalibration
from supermimic_human_ik import align_vector, knee_point, reconstruct_feet
from supermimic_optimizer import acceptance, optimize_motion, quality
from supermimic_physics import G1Physics
from supermimic_policy import SuperMimicPolicy
from supermimic_refinement import residual_candidates
from supermimic_skeleton import G1Skeleton, HumanMotion, HUMAN_TO_G1


def fixture(name: str = 'walk_kit_02') -> dict:
    bank = json.loads((ROOT / 'public/locomotion/bank.json').read_text())
    return next(clip for clip in bank['clips'] if clip['name'] == name)


class GeometryTests(unittest.TestCase):
    def test_coordinate_frame_is_rotation_and_invertible(self):
        np.testing.assert_allclose(HUMAN_TO_G1 @ [0, 1, 0], [0, 0, 1])
        np.testing.assert_allclose(HUMAN_TO_G1 @ [0, 0, 1], [1, 0, 0])
        self.assertAlmostEqual(np.linalg.det(HUMAN_TO_G1), 1)
        r = Rotation.from_euler('xyz', [.4, -.6, .9]).as_matrix()
        np.testing.assert_allclose(HUMAN_TO_G1.T @ (HUMAN_TO_G1 @ r @ HUMAN_TO_G1.T) @ HUMAN_TO_G1, r)

    def test_roundtrip_sampling_preserves_original_duration(self):
        motion = HumanMotion.load(fixture())
        count = int(np.ceil((len(motion.root) - 1) / 20 * 50)) + 1
        output = motion.resample(50, count).resample(20, len(motion.root))
        self.assertEqual(output.document['frames'], motion.document['frames'])
        np.testing.assert_allclose(output.root[0], motion.root[0], atol=1e-8)
        np.testing.assert_allclose(output.root[-1], motion.root[-1], atol=.004)

    def test_invalid_skeleton_fails_explicitly(self):
        for change in [{'up': 'zup'}, {'parents': [-1] * 27}, {'restJoints': [0] * 12},
                       {'restJoints': [0] * 81}, {'frames': True}, {'fps': True}, {'bones': None}]:
            data = fixture(); data.update(change)
            with self.assertRaises(ValueError):
                HumanMotion.load(data)

    def test_resampling_drops_stale_contacts(self):
        motion = HumanMotion.load(fixture())
        output = motion.resample(50)
        self.assertNotIn('footContacts', output.document)

    def test_zero_vector_alignment_rejected(self):
        with self.assertRaises(ValueError):
            align_vector(np.zeros(3), np.ones(3))

    def test_vector_alignment_parallel_antiparallel(self):
        for target in ([0, -1, 0], [0, 1, 0], [1, 0, 0]):
            r = align_vector(np.array([0., -1, 0]), np.array(target, dtype=float))
            np.testing.assert_allclose(r @ [0, -1, 0], target, atol=1e-8)
            self.assertAlmostEqual(np.linalg.det(r), 1)

    def test_two_bone_reach_and_unreachable(self):
        hip, ankle, guide = np.array([0., 1, 0]), np.array([0., .3, .2]), np.array([0., .6, .25])
        knee, excess = knee_point(hip, ankle, guide, .45, .44)
        self.assertAlmostEqual(np.linalg.norm(knee - hip), .45)
        self.assertAlmostEqual(np.linalg.norm(knee - ankle), .44)
        self.assertEqual(excess, 0)
        _, excess = knee_point(hip, np.array([0., -1, 0]), guide, .45, .44)
        self.assertGreater(excess, 1)

    def test_leg_reconstruction_identity(self):
        human = HumanMotion.load(fixture())
        feet = human.positions[:, [human.semantic[side + 'Foot'] for side in ('left', 'right')]] + human.root[:, None]
        local, report = reconstruct_feet(human, feet)
        before = Rotation.from_rotvec(human.local.reshape(-1, 3)).as_matrix()
        after = Rotation.from_rotvec(local.reshape(-1, 3)).as_matrix()
        np.testing.assert_allclose(after, before, atol=1e-6)
        self.assertLess(report['footTargetMaxErrorM'], 1e-6)


class PhysicalBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = SuperMimicPolicy()

    def test_scaled_skeletons_produce_same_g1_reference(self):
        original = fixture('idle')
        doubled = copy.deepcopy(original)
        for key in ('restJoints', 'rootDelta', 'rootOrigin'):
            doubled[key] = (np.array(doubled[key]) * 2).tolist()
        a = adapt_human(HumanMotion.load(original), self.policy)
        b = adapt_human(HumanMotion.load(doubled), self.policy)
        self.assertAlmostEqual(a.skeleton.scale, 2 * b.skeleton.scale)
        np.testing.assert_allclose(a.reference.position, b.reference.position, atol=1e-7)
        np.testing.assert_allclose(a.reference.joints, b.reference.joints, atol=1e-7)

    def test_reverse_without_physics_residual_is_identity(self):
        source = fixture('start_kit_02')
        adapted = adapt_human(HumanMotion.load(source), self.policy)
        ref = adapted.reference
        output = reverse_human(adapted, {'root': ref.position, 'rotation': ref.rotation, 'joints': ref.joints})
        self.assertEqual(output['bones'], source['bones'])
        self.assertEqual(output['parents'], source['parents'])
        self.assertEqual(output['restJoints'], source['restJoints'])
        np.testing.assert_allclose(output['rootDelta'], source['rootDelta'], atol=1e-8)
        before = Rotation.from_rotvec(np.array(source['localRot']).reshape(-1, 3))
        after = Rotation.from_rotvec(np.array(output['localRot']).reshape(-1, 3))
        self.assertLess(np.linalg.norm((before.inv() * after).as_rotvec(), axis=1).max(), 1e-8)

    def test_incomplete_rollout_cannot_replace_source(self):
        adapted = adapt_human(HumanMotion.load(fixture('idle')), self.policy)
        with self.assertRaises(ValueError):
            reverse_human(adapted, {'root': np.zeros((1, 3))})

    def test_real_pretrained_human_walk_pipeline(self):
        source = fixture()
        saved = copy.deepcopy(source)
        result = optimize_motion(source, self.policy)
        report, candidate = result['report'], result['candidate']
        self.assertTrue(report['tracking']['completed'])
        self.assertIsNotNone(candidate)
        self.assertEqual(source, saved)
        self.assertEqual(candidate['frames'], source['frames'])
        self.assertEqual(candidate['restJoints'], source['restJoints'])
        self.assertLess(report['after']['supportMean'], report['before']['supportMean'])
        self.assertLess(report['rootErrorMaxM'], .12)
        self.assertLess(report['reconstruction']['footTargetMaxErrorM'], .025)
        self.assertFalse(report['avatarValidated'])
        if not report['accepted']:
            self.assertEqual(result['motion'], source)


class GateTests(unittest.TestCase):
    def test_regressions_do_not_pass_gate(self):
        before = quality(fixture())
        after = {**before, 'supportMean': before['supportMean'] * .8}
        self.assertEqual(acceptance(before, after, .05, {'footTargetMaxErrorM': 0}), [])
        for change in [{'supportP95': 1}, {'supportSamples': 0}, {'angularStepMax': 1},
                       {'pairedSupportMean': 1}, {'pairedSupportP95': 1}]:
            self.assertTrue(acceptance(before, {**after, **change}, .05, {'footTargetMaxErrorM': 0}))
        self.assertTrue(acceptance(before, after, .5, {'footTargetMaxErrorM': 0}))


class AvatarTests(unittest.TestCase):
    def setUp(self):
        self.source = HumanMotion.load(fixture())
        self.profile = {'version': 1, 'avatar': 'test-avatar', 'positions': {
            name: (self.source.rest[index] * .8).tolist() for name, index in self.source.semantic.items()}}

    def test_collapsed_semantics_preserve_rotations_and_roundtrip(self):
        calibration = AvatarCalibration.load(self.source, self.profile, .8)
        self.assertEqual(calibration.human.document['bones'].count('chest'), 1)
        np.testing.assert_allclose(calibration.human.world, self.source.world[:, calibration.source_indices], atol=1e-10)
        np.testing.assert_allclose(calibration.human.rest, self.source.rest[calibration.source_indices] * .8, atol=1e-10)
        output = calibration.restore(calibration.human.document)
        np.testing.assert_allclose(output['localRot'], self.source.document['localRot'], atol=1e-10)
        np.testing.assert_allclose(output['rootDelta'], self.source.document['rootDelta'], atol=1e-10)

    def test_missing_scale_and_invalid_profile_rejected(self):
        with self.assertRaises(ValueError):
            AvatarCalibration.load(self.source, self.profile)
        for scale in (True, float('nan'), 0):
            with self.assertRaises(ValueError):
                AvatarCalibration.load(self.source, self.profile, scale)
        del self.profile['positions']['leftFoot']
        with self.assertRaises(ValueError):
            AvatarCalibration.load(self.source, self.profile, .8)

    def test_residual_smoothing_preserves_entry_and_source(self):
        document = copy.deepcopy(self.source.document)
        root = self.source.root.copy(); root[1:, 1] += .03
        document.update(rootDelta=root.ravel().tolist(), supermimicReconstruction={'footMode': 'residual'})
        saved = copy.deepcopy(self.source.document)
        candidates = list(residual_candidates(self.source, document))
        self.assertEqual(len(candidates), 16)
        for output, settings in candidates:
            np.testing.assert_allclose(output['rootDelta'][:3], self.source.root[0], atol=1e-10)
            self.assertLessEqual(settings['strength'], 1)
            np.testing.assert_allclose(output['localRot'], self.source.document['localRot'], atol=1e-10)
        self.assertEqual(saved, self.source.document)


if __name__ == '__main__':
    unittest.main()
