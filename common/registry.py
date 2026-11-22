"""Registry {name: ReasoningModule}. Entries are HTTP proxies built from services.yaml.

Adding a module = adding a `kind: module` entry to services.yaml. No UI changes needed.
"""
from pathlib import Path
from typing import Optional

from common.client import HttpReasoningModule
from common.services import load_services


def build_registry(services_path: Optional[Path] = None) -> dict[str, HttpReasoningModule]:
    return {
        spec.display_name or spec.name: HttpReasoningModule(spec.display_name or spec.name, spec.base_url)
        for spec in load_services(services_path).values()
        if spec.kind == "module"
    }
