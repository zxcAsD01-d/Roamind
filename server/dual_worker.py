"""A separate Python execution context for the second resident ARDY model."""
from __future__ import annotations

import multiprocessing
import atexit
import os
import time
import traceback


def worker_main(connection):
    from realtime import RealtimeRunner
    from dual import DualRunner
    try:
        runner = RealtimeRunner()
        connection.send({"ready": True, "pid": os.getpid()})
        while True:
            request = connection.recv()
            if request["kind"] == "close":
                return
            try:
                with runner.torch.inference_mode():
                    if request["kind"] == "warmup":
                        runner.warmup()
                        connection.send({"ok": True})
                    else:
                        window = infer(runner, request, DualRunner)
                        connection.send({"window": window})
            except Exception:
                connection.send({"error": traceback.format_exc()})
    except EOFError:
        return
    except Exception:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


def infer(runner, request, implementation):
    started = time.perf_counter()
    local, root = request["local"], request["root"]
    history = runner._encode_history(local, root) if len(root) else None
    if request["plan"] is None:
        window = implementation._free_window(runner, history, request["prompt"], local, root, request.get('pending'))
    else:
        window = implementation._plan_window(runner, history, request["start"],
            request["origin"], request["plan"], lambda: False, request.get('pending'))
    window.stats.update({"lane": 1, "startFrame": request["start"], "pid": os.getpid(),
                         "startedMs": started*1000, "finishedMs": time.perf_counter()*1000})
    return window


class ModelProcess:
    def __init__(self):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker_main, args=(child,), name="ardy-model-B", daemon=True)
        self.process.start()
        child.close()
        try:
            ready = self.receive(timeout=60)
            if not ready.get("ready"):
                raise RuntimeError(f"第二模型加载失败：{ready}")
        except Exception:
            self.close()
            raise
        atexit.register(self.close)

    def close(self):
        self.connection.close()
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)

    def receive(self, timeout=30):
        deadline = time.monotonic()+timeout
        try:
            while not self.connection.poll(.05):
                if not self.process.is_alive():
                    raise RuntimeError(f"第二模型进程已退出（{self.process.exitcode}）")
                if time.monotonic() > deadline:
                    raise TimeoutError("第二模型推理超时，已终止进程；请重启动作服务")
            result = self.connection.recv()
        except (EOFError, OSError, TimeoutError, RuntimeError):
            self.close()
            raise
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def warmup(self):
        self.connection.send({"kind": "warmup"})
        self.receive(timeout=60)

    def infer(self, local, root, prompt, start, origin, plan, pending=None):
        self.connection.send({"kind": "infer", "local": local, "root": root,
                              "prompt": prompt, "start": start, "origin": origin, "plan": plan,
                              'pending': pending})
        return self.receive()["window"]
