"""List the models your API key can call through the OpenAI-compatible endpoint, then check AI Studio
(aistudio.google.com -> your key -> rate limits) for which of them are free and their daily caps.

    python modules/pgr/scripts/list_models.py
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from modules.pgr.config import LLMConfig  # noqa: E402


def main() -> None:
    cfg = LLMConfig()
    key = os.environ.get(cfg.api_key_env)
    if not key:
        raise SystemExit(f"Set {cfg.api_key_env} first.")
    url = cfg.url.rsplit("/chat/completions", 1)[0] + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    ids = sorted(m.get("id", "") for m in data.get("data", []))
    print("\n".join(i.removeprefix("models/") for i in ids))


if __name__ == "__main__":
    main()
