"""Real local vision/ASR tests; public CV samples and macOS synthesized speech."""
import base64
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"output/performance"
BASE="http://127.0.0.1:8200"


def post(path,body):
    req=urllib.request.Request(BASE+path,json.dumps(body).encode(),{"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=90) as response:return json.load(response)
    except urllib.error.HTTPError as exc:
        raise AssertionError(f"{path}: {exc.read().decode()}") from exc


def sample(name,url):
    path=OUTPUT/name
    if not path.exists():
        with urllib.request.urlopen(url,timeout=30) as response:path.write_bytes(response.read())
    return path


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    bus=sample("vision-test.jpg","https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg")
    face=sample("face-test.jpg","https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg")
    report={}
    for label,path in (("yolo",bus),("face",face)):
        report[label]=post("/api/vision",{"image":base64.b64encode(path.read_bytes()).decode()})
    assert any(obj["label"]=="person" for obj in report["yolo"]["objects"]),report
    assert report["face"]["faces"] and report["face"]["faces"][0]["identity"] is None,report
    wave=OUTPUT/"asr-test.wav"
    if not wave.exists():
        subprocess.run(["say","-v","Tingting","你好，欢迎来到动作研究室。请面向前方，然后慢慢走到餐厅。","-o",str(OUTPUT/"asr-test.aiff")],check=True)
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(OUTPUT/"asr-test.aiff"),"-ar","16000","-ac","1",str(wave)],check=True)
    signal,rate=sf.read(wave)
    audio={"pcm16":base64.b64encode((np.clip(signal,-1,1)*32767).astype("<i2").tobytes()).decode(),
           "sampleRate":rate,"channels":1,"language":"zh"}
    report["audio"]=post("/api/audio",audio)
    warm=post("/api/audio",audio)
    assert "餐厅" in warm["text"],warm
    assert warm["speaker"]["anonymousId"]==report["audio"]["speaker"]["anonymousId"]
    assert not warm["direction"]["available"]
    assert report["audio"]["speaker"]["identity"] is None
    report["warmAudio"]=warm
    report["timestamp"]=time.time()
    (OUTPUT/"perception-integration.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({"detectedPeople":len(report["yolo"]["objects"]),"detectedFaces":len(report["face"]["faces"]),
                      "text":warm["text"],"warmAudioMs":warm["inferenceMs"],"monoDirectionRefused":True},ensure_ascii=False))


if __name__=="__main__":main()
