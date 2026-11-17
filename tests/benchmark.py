"""Reproducible resident inference/navigation benchmark (HTTP service required)."""
import json
import platform
import statistics
import time
import urllib.request
import urllib.error
import uuid
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

BASE = "http://127.0.0.1:8100"
ROOT = Path(__file__).resolve().parents[1]


def post(path, body):
    try:
        return urllib.request.urlopen(urllib.request.Request(BASE+path,json.dumps(body).encode(),
            {"Content-Type":"application/json"}),timeout=60)
    except urllib.error.HTTPError as exc:
        raise AssertionError(f"{path}: {exc.read().decode()}") from exc


def percentile(values, fraction):
    return sorted(values)[min(len(values)-1,int(len(values)*fraction))]


def stream(plan_id=None, duration=12):
    started = time.perf_counter()
    chunks, first = [], None
    with post('/api/generate/stream',{"clientId":uuid.uuid4().hex,"revision":1,
        "prompt":"A person walks forward.","duration":duration,**({"planId":plan_id} if plan_id else {})}) as response:
        for line in response:
            event=json.loads(line)
            if event['type']=='error':
                raise AssertionError(event)
            if event['type']=='chunk':
                if first is None:
                    first=time.perf_counter()-started
                chunks.append(event)
            if event['type']=='complete':
                entry=event['entry']
    elapsed=time.perf_counter()-started
    times=[c['generationMs'] for c in chunks]
    frames=sum(c['frames'] for c in chunks)
    return {"firstChunkMs":round(first*1000,2),"totalMs":round(elapsed*1000,2),
            "motionSeconds":frames/20,"realtimeFactor":round(elapsed/(frames/20),3),
            "windowP50Ms":round(statistics.median(times),2),"windowP95Ms":percentile(times,.95),
            "chunks":len(chunks),"entry":entry,
            "engine": chunks[-1]["engine"], "continuity": continuity(chunks),
            "rootProjectionMax":max((c.get('constraintStats',{}).get('rootProjectionMax',0) for c in chunks),default=0),
            "refinementPasses":sum(c.get('constraintStats',{}).get('refinementPasses',0) for c in chunks)}


def continuity(chunks):
    local = np.concatenate([np.asarray(c["localRot"]).reshape(-1, 27, 3) for c in chunks])
    roots = np.concatenate([np.asarray(c["rootDelta"]).reshape(-1, 3) for c in chunks])
    previous = Rotation.from_rotvec(local[:-1].reshape(-1, 3))
    current = Rotation.from_rotvec(local[1:].reshape(-1, 3))
    steps = (previous.inv()*current).magnitude().reshape(-1, 27)
    boundaries = np.cumsum([c["frames"] for c in chunks[:-1]]) - 1
    return {"boundaryMaxRadians": round(float(steps[boundaries].max()), 6),
            "jointStepP95Radians": round(float(np.percentile(steps, 95)), 6),
            "jointStepMaxRadians": round(float(steps.max()), 6),
            "rootAccelerationP95": round(float(np.percentile(np.linalg.norm(np.diff(roots, n=2, axis=0), axis=1), 95)), 6)}


def main():
    with urllib.request.urlopen(BASE+'/health') as response:
        engine = json.load(response)["engine"]
    routes=[]
    for destination in ['bedroom1','bedroom2','bedroom3','bedroom4','dining','kitchen','bathroom1','bathroom2','bathroom3']:
        with post('/api/plans/navigation',{'destination':destination}) as response:
            plan=json.load(response)
        assert 'error' not in plan,plan
        routes.append(plan['planningMs'])
    with post('/api/plans/timeline',{'steps':[
        {'action':'walk','start':0,'end':14,'destination':'dining'},
        {'action':'wave','start':14,'end':18}]}) as response:
        plan=json.load(response)
    with post('/api/plans/navigation',{'destination':'bedroom4'}) as response:
        bedroom=json.load(response)
    result={"platform":platform.platform(),"timestamp":time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            "backend":f"PyTorch {engine['device']}, dual process, DDIM 6 steps", "routeP50Ms":statistics.median(routes),
            "routeP95Ms":percentile(routes,.95),"free":stream(),"houseTimeline":stream(plan['id']),
            "bedroomDoorway":stream(bedroom['id'])}
    assert result['free']['realtimeFactor']<1, result
    assert result['houseTimeline']['realtimeFactor']<1, result
    assert result['bedroomDoorway']['realtimeFactor']<1, result
    out=ROOT/'output/performance/service-benchmark.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
