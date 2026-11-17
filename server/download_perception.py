"""Download official pretrained models into the isolated perception workspace."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT=Path(__file__).resolve().parents[1]
MODELS=ROOT/"models/perception"
SOURCES={
    'yolo11s-pose.pt':'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s-pose.pt',
    'yolo11n-pose.pt':'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt',
    "yolov8s-worldv2.pt":"https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt",
    "face_detection_yunet.onnx":"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface.onnx":"https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def download(name,url):
    path=MODELS/name
    if path.exists() and path.stat().st_size>10000:
        return path
    temporary=path.with_suffix(".download")
    try:
        with urllib.request.urlopen(url,timeout=180) as response,temporary.open("wb") as output:
            while block:=response.read(1024*1024):output.write(block)
        if temporary.stat().st_size<10000:
            raise ValueError(f"下载文件过小：{name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


if __name__=="__main__":
    MODELS.mkdir(parents=True,exist_ok=True)
    for name,url in SOURCES.items():
        print(download(name,url),flush=True)
    from ultralytics import settings
    settings.update({"weights_dir":str(MODELS)})
    import clip
    clip.load("ViT-B/32",device="cpu",download_root=str(MODELS/"clip"))
    snapshot_download("mlx-community/whisper-base-mlx",local_dir=MODELS/"whisper-base-mlx",
                      allow_patterns=["config.json","weights.npz"])
    manifest=[{"file":str(path.relative_to(MODELS)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
              for path in MODELS.rglob("*") if path.is_file() and path.suffix in (".pt",".onnx",".npz")]
    (MODELS/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("感知模型准备完成")
