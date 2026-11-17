"""YOLO-World + YuNet/SFace, MLX Whisper and local anonymous voice embeddings."""
import base64
import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from audio_direction import direction

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT/"models/perception"
DEFAULT_CLASSES = ["person", "face", "chair", "sofa", "table", "bed", "door", "refrigerator",
                   "floor", "wall", "window", "cabinet", "remote control", "computer monitor", "keyboard",
                   "computer mouse", "sandwich", "cup", "water dispenser", "bottle"]


def image_data(args):
    raw = str(args.get("image", "")).split(",")[-1]
    if len(raw) > 3_000_000:
        raise ValueError("图像超过 2 MB 限制")
    data = np.frombuffer(base64.b64decode(raw,validate=True),np.uint8)
    image = cv2.imdecode(data,cv2.IMREAD_COLOR)
    if image is None or max(image.shape[:2]) > 1920:
        raise ValueError("需要有效 JPEG/PNG，最长边不超过1920")
    return image


class Vision:
    def __init__(self, identities):
        import torch
        from ultralytics import YOLOWorld, YOLO
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        self.lock = threading.Lock()
        self.identities = identities
        self.model = YOLOWorld(str(MODELS/"yolov8s-worldv2.pt"))
        self.pose = YOLO(os.environ.get('MOTION_POSE_MODEL', str(MODELS/'yolo11s-pose.pt')))
        self.classes = []
        self.face = cv2.FaceDetectorYN.create(str(MODELS/"face_detection_yunet.onnx"),"",(640,360),.8,.3,500)
        self.recognizer = cv2.FaceRecognizerSF.create(str(MODELS/"face_recognition_sface.onnx"),"")
        self._classes(DEFAULT_CLASSES)
        self.model.predict(np.zeros((360,640,3),np.uint8),device="cpu",imgsz=640,verbose=False)

    def _classes(self, classes):
        if not isinstance(classes,list) or not 1 <= len(classes) <= 32:
            raise ValueError("classes 需要1–32条英文类别文本")
        if any(not isinstance(c,str) or not 1 <= len(c) <= 80 for c in classes):
            raise ValueError("类别必须是1–80字符的文本")
        if classes != self.classes:
            self.model.set_classes(classes)
            self.classes = classes.copy()

    def infer(self, args):
        started = time.perf_counter()
        image = image_data(args)
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("视觉推理忙，跳过旧帧")
        try:
            self._classes(args.get("classes",DEFAULT_CLASSES))
            prediction = self.model.predict(image,conf=.20,imgsz=640,device="cpu",verbose=False,max_det=32)[0]
            objects = [{"label":prediction.names[int(box.cls.item())],"confidence":round(float(box.conf.item()),3),
                        "box":box.xyxy[0].tolist()} for box in prediction.boxes]
            from pose_depth import decode_poses
            people = decode_poses(self.pose.predict(image, conf=.25, imgsz=640, device='cpu', verbose=False, max_det=8)[0]) if args.get('pose') else []
            faces = self._faces(image) if args.get("faces",True) else []
        finally:
            self.lock.release()
        return {"objects":objects,"faces":faces,'people': people,"width":image.shape[1],"height":image.shape[0],
                "observedAt":args.get("observedAt",time.time()),"inferenceMs":round((time.perf_counter()-started)*1000,2),
                "source":"ego-yolo", "model":"yolov8s-worldv2", "identityAssociation":"unverified"}

    def _faces(self, image):
        self.face.setInputSize((image.shape[1],image.shape[0]))
        _, detected = self.face.detect(image)
        result = []
        for face in ([] if detected is None else detected[:8]):
            crop = self.recognizer.alignCrop(image,face)
            vector = self.recognizer.feature(crop).reshape(-1)
            x,y,w,h = [float(v) for v in face[:4]]
            result.append({"box":[x,y,x+w,y+h],"confidence":float(face[-1]),
                           'landmarks': face[4:14].reshape(5, 2).astype(float).tolist(),
                           **self.identities.remember("face",vector)})
        return result


def audio_data(args):
    rate = int(args.get("sampleRate",16000))
    channels = int(args.get("channels",1))
    if not 8000 <= rate <= 96000 or channels not in (1,2):
        raise ValueError("PCM16 采样率8000–96000，只支持1/2声道")
    encoded = args.get("pcm16", "")
    if not isinstance(encoded,str) or len(encoded) > 4_200_000:
        raise ValueError("音频数据无效或过大")
    raw = base64.b64decode(encoded,validate=True)
    if len(raw) % (2*channels):
        raise ValueError("PCM16 必须按帧交错排列")
    data = np.frombuffer(raw,dtype="<i2").astype(np.float32).reshape(-1,channels)/32768
    if not .5 <= len(data)/rate <= 12:
        raise ValueError("每段音频需0.5–12秒")
    return data, rate


class Audio:
    def __init__(self, identities):
        from resemblyzer import VoiceEncoder
        self.voice = VoiceEncoder(device="cpu",verbose=False).eval()
        self.identities = identities
        self.lock = threading.Lock()
        self.whisper_model = str(MODELS/"whisper-base-mlx")
        if not (Path(self.whisper_model)/"config.json").exists():
            raise FileNotFoundError("请先下载 Whisper base MLX 权重")

    def infer(self, args):
        samples, rate = audio_data(args)
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("音频推理忙，请稍后提交")
        try:
            return self._infer(samples,rate,args)
        finally:
            self.lock.release()

    def _infer(self, samples, rate, args):
        import mlx_whisper
        from resemblyzer import preprocess_wav
        from scipy.signal import resample_poly
        started = time.perf_counter()
        mono = samples.mean(axis=1)
        if np.sqrt(np.mean(mono**2)) < .003:
            return {"text":"","segments":[],"speaker":None,"direction":{"available":False,"reason":"静音"},"silence":True}
        divisor = math.gcd(rate,16000)
        mono = resample_poly(mono,16000//divisor,rate//divisor).astype(np.float32)
        speech = preprocess_wav(mono,source_sr=16000)
        if len(speech) < 8000:
            return {"text":"","segments":[],"speaker":None,"direction":{"available":False,"reason":"有效语音不足"},"silence":True}
        result = mlx_whisper.transcribe(mono,path_or_hf_repo=self.whisper_model,
                    language=args.get("language") or None,fp16=True,verbose=None,
                    condition_on_previous_text=False,temperature=0.,word_timestamps=True)
        segments = [{"start":s["start"],"end":s["end"],"text":s["text"]} for s in result["segments"]
                    if s.get("no_speech_prob",0) < .6]
        voice = self.identities.remember("speaker",self.voice.embed_utterance(speech))
        return {"text":"".join(s["text"] for s in segments),"segments":segments,"speaker":voice,
                "direction":direction(samples,rate,args.get("microphoneSpacing"),args.get("arrayYaw",0)),
                "observedAt":args.get("observedAt",time.time()),"duration":len(mono)/16000,
                "inferenceMs":round((time.perf_counter()-started)*1000,2),"asr":"mlx-whisper-base",
                "speakerScope":"one-dominant-speaker-per-segment", "faceSpeakerAssociation":"unverified"}
