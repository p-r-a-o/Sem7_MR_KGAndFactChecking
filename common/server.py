"""Tiny stdlib JSON-over-HTTP server so each module's api.py stays a thin wrapper.

`serve(routes, port)` takes {(METHOD, path): handler(body_dict) -> dict}.
`module_routes(module, load_kg)` builds the standard /health /load /predict_link /verify_claim routes.
Requests are handled one at a time (a lock), because model/torch state is not thread-safe.
"""
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from common.api import ReasoningModule
from common.datasets.base import KnowledgeGraph
from common.errors import ModuleError
from common.serialization import to_dict

Handler = Callable[[dict], dict]
Routes = dict[tuple[str, str], Handler]


def _make_handler_class(routes: Routes, lock: threading.Lock):
    class _Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _dispatch(self, method: str) -> None:
            handler = routes.get((method, self.path.split("?")[0]))
            if handler is None:
                return self._send(404, {"error": {"code": "BAD_REQUEST", "message": f"No route {method} {self.path}"}})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}") if length else {}
                with lock:
                    result = handler(body)
                self._send(200, result)
            except ModuleError as e:
                self._send(e.status, e.to_body())
            except json.JSONDecodeError:
                self._send(400, ModuleError("BAD_REQUEST", "Body is not valid JSON").to_body())
            except Exception as e:  # noqa: BLE001 - last-resort uniform error body
                traceback.print_exc()
                self._send(500, ModuleError("INTERNAL", f"{type(e).__name__}: {e}").to_body())

        def do_GET(self):  # noqa: N802
            self._dispatch("GET")

        def do_POST(self):  # noqa: N802
            self._dispatch("POST")

        def log_message(self, fmt, *args):  # quieter logs
            print(f"[{self.log_date_time_string()}] {fmt % args}")

    return _Handler


def serve(routes: Routes, port: int, host: str = "127.0.0.1") -> None:
    server = ThreadingHTTPServer((host, port), _make_handler_class(routes, threading.Lock()))
    print(f"Serving on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _require(body: dict, key: str):
    if body.get(key) is None:
        raise ModuleError("BAD_REQUEST", f"Missing field: {key}")
    return body[key]


def module_routes(module: ReasoningModule, load_kg: Callable[[str], KnowledgeGraph]) -> Routes:
    state: dict[str, Optional[str]] = {"loaded": None}

    def health(_: dict) -> dict:
        return {"name": module.name, "status": "ok", "loaded": state["loaded"]}

    def load(body: dict) -> dict:
        dataset = _require(body, "dataset")
        try:
            kg = load_kg(dataset)
        except (FileNotFoundError, KeyError) as e:
            raise ModuleError("INTERNAL", f"Cannot load dataset {dataset!r}: {e}. Run scripts/download_data.py.")
        module.load(kg)
        state["loaded"] = dataset
        return {"status": "ok"}

    def _need_loaded() -> None:
        if state["loaded"] is None:
            raise ModuleError("NOT_LOADED", "Call POST /load first.")

    def predict_link(body: dict) -> dict:
        _need_loaded()
        head, tail = body.get("head"), body.get("tail")
        if (head is None) == (tail is None):
            raise ModuleError("BAD_REQUEST", "Exactly one of head/tail must be null.")
        return to_dict(module.predict_link(head, _require(body, "relation"), tail, int(body.get("k", 10))))

    def verify_claim(body: dict) -> dict:
        _need_loaded()
        return to_dict(module.verify_claim(_require(body, "claim")))

    return {
        ("GET", "/health"): health,
        ("POST", "/load"): load,
        ("POST", "/predict_link"): predict_link,
        ("POST", "/verify_claim"): verify_claim,
    }
