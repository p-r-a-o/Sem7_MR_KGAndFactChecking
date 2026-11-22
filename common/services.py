"""Read services.yaml (stdlib-only, so a tiny parser for the flat subset we use).

Supported: nested `key:` maps by indentation, `key: scalar`, full-line `#` comments.
Not supported: lists, inline comments, multi-line scalars.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from common.paths import repo_root


def _scalar(value: str):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return int(value) if value.isdigit() else value


def parse_simple_yaml(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, value = raw.strip().partition(":")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _scalar(value)
    return root


@dataclass(frozen=True)
class ServiceSpec:
    name: str                 # key in services.yaml and registry id
    kind: str                 # "module" | "service"
    port: int
    host: str = "127.0.0.1"
    display_name: str = ""
    env: str = ""             # environment file (informational)
    start: str = ""           # shell command that starts it

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def load_services(path: Optional[Path] = None) -> dict[str, ServiceSpec]:
    path = Path(path or os.environ.get("KGX_SERVICES_FILE", repo_root() / "services.yaml"))
    raw = parse_simple_yaml(path.read_text(encoding="utf-8")).get("services", {})
    return {
        name: ServiceSpec(
            name=name,
            kind=cfg.get("kind", "service"),
            port=int(cfg["port"]),
            host=cfg.get("host", "127.0.0.1"),
            display_name=cfg.get("display_name", name),
            env=cfg.get("env", ""),
            start=cfg.get("start", ""),
        )
        for name, cfg in raw.items()
    }


def get_service(name: str, path: Optional[Path] = None) -> ServiceSpec:
    return load_services(path)[name]
