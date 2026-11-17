import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"server"))
from identities import IdentityStore


class IdentityTests(unittest.TestCase):
    def test_opt_in_enrol_match_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"identities.json"
            store=IdentityStore(path)
            vector=np.random.default_rng(10).normal(size=256)
            sample=store.remember("speaker",vector)
            self.assertIsNone(sample["identity"])
            args={"name":"synthetic-test","modality":"speaker","sampleId":sample["sampleId"]}
            with self.assertRaises(ValueError):store.enroll(args)
            store.enroll({**args,"consent":True})
            match=store.remember("speaker",vector)
            self.assertEqual(match["identity"],"synthetic-test")
            self.assertEqual(sample["anonymousId"],match["anonymousId"])
            store.delete({"name":"synthetic-test"})
            self.assertIsNone(store.remember("speaker",vector)["identity"])
            self.assertEqual(IdentityStore(path).listing(),[])


if __name__=="__main__":unittest.main(verbosity=2)
