"""Local, isolated SuperMimic inference. Never blocks the existing animation player."""
from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from supermimic_policy import MODEL_DIRECTORY, SuperMimicPolicy
from supermimic_runtime import RealtimePolicy, RequestFailure

LOG = logging.getLogger('supermimic')


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, port: int, runtime: RealtimePolicy):
        self.runtime = runtime
        super().__init__(('127.0.0.1', port), Handler)


class Handler(BaseHTTPRequestHandler):
    server: Server

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(3)

    def reply(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, allow_nan=False, separators=(',', ':')).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path != '/health':
            self.reply(404, {'error': 'Not found'})
            return
        self.reply(200, self.server.runtime.health())

    def body(self) -> dict:
        if self.headers.get('Origin') is not None:
            raise RequestFailure('Use a local backend adapter; direct browser requests are disabled', 403)
        if self.headers.get('Transfer-Encoding'):
            raise RequestFailure('Transfer-Encoding is unsupported')
        if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
            raise RequestFailure('Content-Type must be application/json', 415)
        size = int(self.headers.get('Content-Length', '0'))
        if not 1 <= size <= 65536:
            raise RequestFailure('Request body must be between 1 and 65536 bytes', 413)
        data = self.rfile.read(size)
        if len(data) != size:
            raise RequestFailure('Incomplete body')
        value = json.loads(data)
        if not isinstance(value, dict):
            raise RequestFailure('JSON body must be an object')
        return value

    def do_POST(self) -> None:
        self.close_connection = True
        try:
            if self.path != '/infer':
                raise RequestFailure('Not found', 404)
            self.reply(200, self.server.runtime.infer(self.body()))
        except RequestFailure as error:
            self.reply(error.status, {'error': str(error), 'retainOriginalMotion': True})
        except (ValueError, TypeError, OverflowError) as error:
            self.reply(400, {'error': str(error), 'retainOriginalMotion': True})
        except (TimeoutError, ConnectionError):
            LOG.warning('Client disconnected or timed out', exc_info=True)
        except Exception:
            LOG.exception('SuperMimic inference failed')
            self.reply(500, {'error': 'Inference failed; see local service log', 'retainOriginalMotion': True})

    def log_message(self, format: str, *args: object) -> None:
        if len(args) < 2 or str(args[1]) != '200':
            LOG.info(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8300)
    parser.add_argument('--backend', choices=('mlx', 'torch'), default='torch')
    parser.add_argument('--threads', type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    policy = SuperMimicPolicy(MODEL_DIRECTORY, args.backend, args.threads)
    runtime = RealtimePolicy(policy)
    server = Server(args.port, runtime)
    LOG.info('SuperMimic ready at http://127.0.0.1:%s; backend=%s; G1 only', args.port, args.backend)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        LOG.info('Stopping SuperMimic service')
    finally:
        server.server_close()
        runtime.close()


if __name__ == '__main__':
    main()
