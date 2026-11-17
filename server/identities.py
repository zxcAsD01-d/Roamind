"""Local, opt-in biometric enrolment. Raw images/audio are never persisted."""
import json
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1]/".local-data/identities.json"


class IdentityStore:
    def __init__(self, path=PATH):
        self.path = path
        self.lock = threading.RLock()
        self.samples = {}
        self.anonymous = {"face":[], "speaker":[]}
        self.registered = json.loads(path.read_text()) if path.exists() else {}

    def remember(self, modality, vector):
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        vector /= max(float(np.linalg.norm(vector)), 1e-8)
        sample = uuid.uuid4().hex
        threshold = .45 if modality == "face" else .80
        with self.lock:
            self.samples = {key:value for key,value in self.samples.items() if time.time()-value[0] < 120}
            self.samples[sample] = (time.time(), modality, vector)
            if len(self.samples) > 100:
                del self.samples[next(iter(self.samples))]
            matches = [(float(np.dot(vector, embedding)), name) for name, records in self.registered.items()
                       for embedding in records.get(modality, [])]
            score, name = max(matches, default=(0., None))
            pool = self.anonymous[modality]
            candidate = max(((float(vector @ entry[1]), i) for i,entry in enumerate(pool)), default=(0., -1))
            if candidate[0] < threshold:
                if len(pool) >= 128:
                    pool.pop(0)
                pool.append((f"{modality}-{uuid.uuid4().hex[:8]}",vector))
                index = len(pool)-1
            else:
                index = candidate[1]
            return {"sampleId":sample, "identity":name if score >= threshold else None,
                    "anonymousId":pool[index][0], "similarity":round(score,4),
                    "registeredMatch":score >= threshold, "threshold":threshold}

    def enroll(self, args):
        name = str(args.get("name", "")).strip()
        if args.get("consent") is not True:
            raise ValueError("登记人脸/声纹必须明确 consent=true")
        if not 1 <= len(name) <= 80 or args.get("modality") not in self.anonymous:
            raise ValueError("提供姓名及 face/speaker 类型")
        with self.lock:
            sample = self.samples.get(args.get("sampleId"))
            if not sample or time.time()-sample[0] > 120 or sample[1] != args["modality"]:
                raise ValueError("样本已过期或类型不符，请重新采集")
            records = self.registered.setdefault(name, {}).setdefault(sample[1], [])
            records.append(sample[2].tolist())
            del records[:-5]
            self._persist()
            return {"name":name, "modality":sample[1], "samples":len(records)}

    def delete(self, args):
        with self.lock:
            removed = self.registered.pop(args.get("name"), None) is not None
            self._persist()
            return {"deleted":removed}

    def _persist(self):
        self.path.parent.mkdir(exist_ok=True, mode=0o700)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.registered,ensure_ascii=False))
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)

    def listing(self):
        with self.lock:
            return [{"name":name,"modalities":list(records)} for name,records in self.registered.items()]
