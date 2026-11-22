"""HTTP client proxies (stdlib-only). The UI and eval talk to modules only through these."""
import json
import urllib.error
import urllib.request
from typing import Any, Optional

from common.datasets.base import KnowledgeGraph
from common.errors import ModuleError
from common.schema import CVResult, LPResult
from common.serialization import cv_result_from_dict, lp_result_from_dict


def request_json(method: str, url: str, body: Optional[dict] = None, timeout: float = 600.0) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read() or b"{}").get("error", {})
        except json.JSONDecodeError:
            err = {}
        raise ModuleError(err.get("code", "INTERNAL"), err.get("message", f"HTTP {e.code} from {url}"))
    except urllib.error.URLError as e:
        raise ModuleError("INTERNAL", f"Cannot reach {url}: {e.reason}")


class HttpReasoningModule:
    """ReasoningModule that forwards each call to a module's api.py."""

    def __init__(self, name: str, base_url: str, timeout: float = 600.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        return request_json("GET", f"{self.base_url}/health", timeout=10.0)

    def load(self, kg: KnowledgeGraph) -> None:
        request_json("POST", f"{self.base_url}/load", {"dataset": kg.name}, self.timeout)

    def predict_link(self, head: Optional[str], relation: str,
                     tail: Optional[str], k: int = 10) -> LPResult:
        body = {"head": head, "relation": relation, "tail": tail, "k": k}
        return lp_result_from_dict(request_json("POST", f"{self.base_url}/predict_link", body, self.timeout))

    def verify_claim(self, claim: str) -> CVResult:
        return cv_result_from_dict(
            request_json("POST", f"{self.base_url}/verify_claim", {"claim": claim}, self.timeout))


def parse_claim(base_url: str, claim: str, dataset: str, timeout: float = 120.0) -> dict:
    """Call the shared claim parser service. Returns {"triple": [h, r, t], "linked_entities": {...}}.

    Raises ModuleError(UNPARSEABLE_CLAIM | ...) on failure.
    """
    out = request_json("POST", f"{base_url.rstrip('/')}/parse_claim",
                       {"claim": claim, "dataset": dataset}, timeout)
    if not out.get("triple"):
        raise ModuleError("UNPARSEABLE_CLAIM", f"Could not parse claim: {claim!r}")
    return out
