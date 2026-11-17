import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from visual_person_tracking import associate_person


def person(identity, x, stamp):
    return {'id': identity, 'worldPosition': [x, 0, 3.], 'distance': 3., 'observedAt': stamp}


class NavigationPersonTrackingTests(unittest.TestCase):
    def test_moving_person_retains_navigation_id_without_mutating_contact_id(self):
        previous = person('first', 0, 100.)
        candidate = person('new-body-id', .7, 101.)
        result = associate_person([dict(candidate)], previous, 101.)
        self.assertEqual(result['id'], 'first')
        self.assertEqual(result['observationId'], 'new-body-id')
        self.assertEqual(candidate['id'], 'new-body-id')
        repeated = associate_person([dict(candidate)], result, 101.)
        self.assertEqual(repeated['id'], 'first')

    def test_ambiguous_people_and_impossible_speed_are_not_aliased(self):
        previous = person('first', 0, 100.)
        self.assertIsNone(associate_person([person('a', .6, 101), person('b', .7, 101)], previous, 101))
        self.assertIsNone(associate_person([person('a', 3., 101)], previous, 101))

    def test_expired_track_acquires_a_new_id_and_does_not_claim_identity(self):
        result = associate_person([person('new', .1, 104)], person('old', 0, 100), 104)
        self.assertEqual(result['id'], 'new')

    def test_velocity_prediction_prefers_continuation_over_crossing_person(self):
        previous = {**person('first', 0, 100), 'navigationVelocity': [1., 0.]}
        result = associate_person([person('a', -.1, 101), person('b', 1., 101)], previous, 101)
        self.assertEqual(result['observationId'], 'b')


if __name__ == '__main__':
    unittest.main()
