"""Bounded local perception HTTP service, independent of the ARDY model lock."""
import argparse
import copy
import json
import logging
import threading
import time
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from identities import IdentityStore
from perception_models import Audio, Vision
from stereo_service import STEREO
from ocr import TextRecognizer
from ocr_targets import TextTargets
from visual_targets import VisualTargets
from observation_pipeline import ObservationPipeline

LOGGER = logging.getLogger("perception")
STATE_LOCK = threading.Lock()
STATE = {"vision":{"status":"warming"},"audio":{"status":"warming"},"latestVision":None,"latestAudio":None}
MODELS = {}
IDENTITIES = IdentityStore()
CAPACITY = threading.BoundedSemaphore(4)
OCR = TextRecognizer()
TEXT_TARGETS = TextTargets(STEREO, OCR)
VISUAL_TARGETS = VisualTargets(STEREO)
OBSERVATIONS = ObservationPipeline()


def resolve_target(body):
    token = body.get('targetId', '')
    if isinstance(token, str) and token.startswith('text-'):
        return TEXT_TARGETS.resolve(body)
    if isinstance(token, str) and token.startswith('point-'):
        return VISUAL_TARGETS.resolve(body)
    if not token and body.get('pixel') is not None and not body.get('bodyPart'):
        return VISUAL_TARGETS.select(body)
    return STEREO.touch_target(body)


def infer_stereo(body):
    result = STEREO.infer(body)
    VISUAL_TARGETS.update(body['clientId'])
    return result


def reset_stereo(body, release=False):
    result = STEREO.release(body) if release else STEREO.reset(body)
    VISUAL_TARGETS.clear(body['clientId'])
    return result


def warm(name, factory):
    try:
        model = factory(IDENTITIES)
        with STATE_LOCK:
            MODELS[name] = model
            STATE[name] = {"status":"ready"}
    except Exception as exc:
        LOGGER.exception("%s 初始化失败",name)
        with STATE_LOCK:
            STATE[name] = {"status":"error","error":f"{type(exc).__name__}: {exc}"}


def ready_model(name):
    with STATE_LOCK:
        model = MODELS.get(name)
        status = STATE[name].copy()
    if model is None:
        raise RuntimeError(f"{name} 尚未就绪: {status}")
    return model


def fuse_vision(body, result):
    if 'frameId' in body and 'clientId' in body:
        result['semanticFusion'] = STEREO.annotate(body, result['objects'])
        result['people'] = STEREO.pose_points(body, result.get('people', []))
        result['frameId'] = body['frameId']
    return result


def run_model(name, body):
    result = ready_model(name).infer(body)
    if name == 'vision':
        result = fuse_vision(body, result)
    store_result(name, result)
    return result


def store_result(name, result):
    stored = result
    if name == 'vision' and 'semanticFusion' in result:
        stored = {**result, 'semanticFusion': {key: value for key, value in result['semanticFusion'].items()
                                             if key != 'map'}}
    with STATE_LOCK:
        STATE["latestVision" if name == "vision" else "latestAudio"] = stored


def observe(body):
    result = OBSERVATIONS.infer(body, infer_stereo,
                                lambda args: ready_model('vision').infer(args), fuse_vision)
    if result['vision'] is not None:
        store_result('vision', result['vision'])
    return result


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body):
        data = json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin","http://127.0.0.1:5173")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
        self.send_header("Content-Length",str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send(200,{})

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/api/stereo/state", '/api/stereo/frame', '/api/targets/frame', '/api/stereo/planning-map'):
            client_id = parse_qs(parsed.query).get('clientId', [None])[0]
            try:
                if not isinstance(client_id, str):
                    raise ValueError('需要 clientId')
                if parsed.path == '/api/stereo/planning-map':
                    result = STEREO.planning_map(client_id)
                elif parsed.path == '/api/targets/frame':
                    result = VISUAL_TARGETS.frame(client_id)
                else:
                    result = STEREO.body_frame(client_id) if parsed.path.endswith('/frame') else STEREO.snapshot(client_id)
                return self.send(200, result)
            except ValueError as exc:
                return self.send(400, {"error": str(exc)})
            except Exception as exc:
                LOGGER.exception("读取双目状态失败")
                return self.send(503, {"error": f"{type(exc).__name__}: {exc}"})
        if parsed.path not in ("/health","/api/state"):
            return self.send(404,{"error":"not found"})
        with STATE_LOCK:
            result = copy.deepcopy(STATE)
        self.send(200,{**result,"identities":IDENTITIES.listing(),
                       "stereoSessions":STEREO.listing(),"timestamp":time.time()})

    def do_POST(self):
        if not CAPACITY.acquire(blocking=False):
            return self.send(429,{"error":"感知服务繁忙；请丢弃旧帧"})
        try:
            length = int(self.headers.get("Content-Length",0))
            if not 0 < length <= 4_500_000:
                raise ValueError("请求大小无效，最大4.5 MB")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body,dict):
                raise ValueError("需要 JSON 对象")
            handlers = {"/api/vision":lambda:run_model("vision",body),
                        "/api/observe":lambda:observe(body),
                        "/api/ocr":lambda:OCR.infer(body),
                        "/api/ocr/ego":lambda:TEXT_TARGETS.observe(body),
                        "/api/stereo":lambda:infer_stereo(body),
                        "/api/stereo/reset":lambda:reset_stereo(body),
                        "/api/stereo/release":lambda:reset_stereo(body, True),
                        "/api/touch/resolve":lambda:resolve_target(body),
                        "/api/audio":lambda:run_model("audio",body),
                        "/api/identities/enroll":lambda:IDENTITIES.enroll(body),
                        "/api/identities/delete":lambda:IDENTITIES.delete(body)}
            if self.path not in handlers:
                return self.send(404,{"error":"not found"})
            self.send(200,handlers[self.path]())
        except (ValueError, TypeError) as exc:
            self.send(400,{"error":str(exc)})
        except BlockingIOError as exc:
            self.send(429,{"error":str(exc)})
        except (BrokenPipeError,ConnectionResetError):
            LOGGER.info("感知客户端已断开")
        except Exception as exc:
            LOGGER.exception("感知请求失败")
            self.send(503,{"error":f"{type(exc).__name__}: {exc}"})
        finally:
            CAPACITY.release()

    def log_message(self, *_args):
        return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=8200)
    args = parser.parse_args()
    threading.Thread(target=warm,args=("vision",Vision),daemon=True).start()
    threading.Thread(target=warm,args=("audio",Audio),daemon=True).start()
    print(f"Perception service http://127.0.0.1:{args.port}",flush=True)
    ThreadingHTTPServer(("127.0.0.1",args.port),Handler).serve_forever()
