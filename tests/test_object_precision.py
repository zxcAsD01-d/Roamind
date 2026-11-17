"""Position evidence must be stable without inventing metric accuracy."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from affordances import InteractionOptions
from object_precision import position_estimate
from semantic_map import SemanticMap
from stereo_depth import Calibration
import numpy as np


class ObjectPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.options = InteractionOptions()
        self.raw = {'id': 'surface', 'label': 'object', 'worldPosition': [0., 1., 2.],
                    'positionSigmaMeters': .03, 'box': [0, 0, 20, 20], 'confidence': .9,
                    'points': 100, 'surfacePixel': [10, 10]}

    def remember(self, stamp, offset=0., **values):
        self.options._remember({**self.raw, 'worldPosition': [offset, 1., 2.], **values}, int(stamp), stamp)
        return self.options.objects['surface']

    def test_noise_is_fused_but_systematic_uncertainty_does_not_shrink(self):
        for index, offset in enumerate((0., .02, -.01, .01, -.02, 0., .01, -.01, .02)):
            row = self.remember(100+index*.1, offset)
        self.assertLess(abs(row.position[0]), .011)
        self.assertGreaterEqual(row.position_sigma, .03)
        self.assertEqual(len(row.samples), 8)

    def test_outlier_and_repeated_exposure_do_not_refresh_evidence(self):
        self.remember(100)
        self.remember(100.1, .01)
        self.remember(100.2, .6)
        row = self.remember(100.1, .02)
        self.assertEqual(row.observed, 100.1)
        self.assertEqual(row.observations, 2)
        self.assertEqual(len(row.samples), 2)

    def test_precise_new_measurement_can_correct_uncertain_old_measurement(self):
        self.remember(100, positionSigmaMeters=.3)
        row = self.remember(100.1, .4)
        self.assertEqual(row.observed, 100.1)
        self.assertGreaterEqual(row.position_sigma, .15)

    def test_old_history_does_not_freeze_a_later_moved_object(self):
        self.remember(100)
        row = self.remember(104, .5)
        self.assertEqual(row.position[0], .5)
        self.assertEqual(len(row.samples), 1)

    def test_invalid_uncertainty_is_explicit(self):
        for sigma in (0, -1, float('nan'), True):
            with self.assertRaises(ValueError):
                position_estimate({**self.raw, 'positionSigmaMeters': sigma}, 100, None)

    def test_far_stereo_surface_reports_larger_metric_uncertainty(self):
        calibration = Calibration(width=320, height=180, fx=280., fy=280., cx=160., cy=90., baseline=.12)
        near = SemanticMap.position_uncertainty(np.full(100, 1.), calibration)
        far = SemanticMap.position_uncertainty(np.full(100, 4.), calibration)
        self.assertGreater(far, near*7.5)


if __name__ == '__main__':
    unittest.main()
