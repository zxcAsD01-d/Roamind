"""One exposure, parallel depth/detection, then same-frame semantic fusion."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time


class ObservationPipeline:
    def __init__(self):
        self.capacity = threading.BoundedSemaphore(1)
        self.detector = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ego-detection')

    def infer(self, body, infer_depth, infer_vision, fuse):
        if not self.capacity.acquire(blocking=False):
            raise BlockingIOError('同帧感知正在处理，请提交下一张新图像')
        future = None
        started = time.perf_counter()
        try:
            if body.get('includeVision') is True:
                args = {**body, 'image': body.get('left'), 'pose': True}
                future = self.detector.submit(infer_vision, args)
            depth = infer_depth(body)
            vision, vision_error = self.vision_result(future)
            if vision is not None:
                vision = fuse(body, vision)
                fusion = vision.get('semanticFusion', {})
                if fusion.get('map') is not None:
                    depth['map'] = fusion.pop('map')
            return {'depth': depth, 'vision': vision, 'pipeline': {
                'mode': 'same-exposure-parallel', 'elapsedMs': round((time.perf_counter()-started)*1000, 2),
                'ageSeconds': round(max(0., time.time()-body['observedAt']), 3), 'queuedFrames': 0,
                **({'visionError': vision_error} if vision_error else {})}}
        finally:
            # A depth error must not leave an unbounded detector job queue.
            if future is not None and not future.done() and not future.cancel():
                future.add_done_callback(lambda _: self.capacity.release())
            else:
                self.capacity.release()

    @staticmethod
    def vision_result(future):
        if future is None:
            return None, None
        try:
            return future.result(), None
        except (BlockingIOError, RuntimeError, ValueError) as exc:
            return None, f'{type(exc).__name__}: {exc}'

    def close(self):
        self.detector.shutdown(wait=True, cancel_futures=True)
