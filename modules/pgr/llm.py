"""Chat-completions client for a free LLM tier (Gemini's OpenAI-compatible endpoint by default).

Stdlib only. Three things a free tier needs that the reference code (plain requests.post) lacks:
  * a disk cache keyed by (model, prompt, sampling settings): a rerun, a crash or the UI asking the
    same claim again costs no quota, and evaluation can resume the next day;
  * pacing to stay under the requests-per-minute cap, and waiting out a per-minute 429;
  * a clean stop when the DAILY quota is used up (`DailyQuotaExhausted`), instead of retrying forever.
"""
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from modules.pgr.config import LLMConfig


class LLMError(Exception):
    pass


class DailyQuotaExhausted(LLMError):
    """The provider's per-day limit is reached: stop and resume tomorrow (cached replies are kept)."""


def _retry_delay(body: str, headers) -> Optional[float]:
    after = headers.get("Retry-After") if headers else None
    if after and after.strip().isdigit():
        return float(after)
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body) or re.search(r"retry in (\d+(?:\.\d+)?)s", body)
    return float(m.group(1)) if m else None


def is_daily_quota(body: str) -> bool:
    """True only if the 429 names a per-day quota. Gemini lists the violated quotas as quotaId values
    (e.g. GenerateRequestsPerMinutePerProjectPerModel-FreeTier); judge by those when present. Without
    them, a "retry in Ns" hint means a short (per-minute) wait, not a daily stop."""
    ids = re.findall(r'"quotaId"\s*:\s*"([^"]+)"', body)
    if ids:
        return any("PerDay" in i for i in ids)
    if _retry_delay(body, None) is not None:
        return False
    return bool(re.search(r"per ?day|daily", body, re.IGNORECASE))


class ChatClient:
    def __init__(self, cfg: LLMConfig, cache_path: Optional[Path]):
        self.cfg = cfg
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        self.calls = 0            # real API calls made (cache hits excluded)
        self.cache_hits = 0
        self._last = 0.0
        self._lock = threading.Lock()
        if cache_path and cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        self.cache[d["key"]] = d["reply"]
                    except (ValueError, KeyError):
                        continue      # a half-written last line after a crash

    def _key(self, prompt: str) -> str:
        c = self.cfg
        parts = [c.model, prompt, c.temperature, c.top_p] + ([c.reasoning_effort] if c.reasoning_effort else [])
        raw = json.dumps(parts, ensure_ascii=False)       # unchanged key when reasoning_effort is unset
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def complete(self, prompt: str) -> str:
        """One user message -> the reply text. Cached; paced; retried on transient errors."""
        key = self._key(prompt)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        api_key = os.environ.get(self.cfg.api_key_env)
        if not api_key:
            raise LLMError(f"Set the {self.cfg.api_key_env} environment variable to your API key "
                           f"(Google AI Studio -> Get API key). It is read at call time and never stored.")
        system = "You are a helpful assistant."
        if self.cfg.model.lower().startswith("gemma"):
            # Gemma on the Gemini API rejects system ("developer") instructions: send it in the user turn
            messages = [{"role": "user", "content": f"{system}\n\n{prompt}"}]
        else:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        body = {"model": self.cfg.model, "temperature": self.cfg.temperature, "top_p": self.cfg.top_p,
                "max_tokens": self.cfg.max_tokens, "stream": False, "messages": messages}
        if self.cfg.reasoning_effort:
            body["reasoning_effort"] = self.cfg.reasoning_effort
        data = json.dumps(body).encode("utf-8")
        last_err = ""
        for attempt in range(self.cfg.retries + 2):      # +2: room for waiting out per-minute 429s
            self._pace()
            req = urllib.request.Request(self.cfg.url, data=data, method="POST", headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
                    reply = json.loads(resp.read().decode("utf-8"))
                text = (reply.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                self.calls += 1
                self._store(key, text)
                return text
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", "replace")
                last_err = f"HTTP {e.code}: {err_body[:300]}"
                if e.code == 429:
                    if is_daily_quota(err_body):
                        quota = re.search(r'"quotaId"\s*:\s*"([^"]+)"', err_body)
                        limit = re.search(r'"quotaValue"\s*:\s*"?(\d+)', err_body)
                        detail = (f" (provider: {quota.group(1) if quota else 'daily limit'}"
                                  f"{', limit ' + limit.group(1) if limit else ''})")
                        raise DailyQuotaExhausted(
                            f"Daily free-tier quota for {self.cfg.model} is used up{detail}. Rerun tomorrow: "
                            f"finished claims are cached and skipped.") from None
                    time.sleep(min(_retry_delay(err_body, e.headers) or 30.0, 120.0))
                    continue
                if e.code in (400, 401, 403, 404):
                    raise LLMError(last_err) from None       # bad key / model name: retrying won't help
                time.sleep(5)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = repr(e)
                time.sleep(5)
        raise LLMError(f"LLM call failed after retries: {last_err}")

    def _pace(self) -> None:
        with self._lock:
            gap = 60.0 / max(self.cfg.requests_per_minute, 0.1)
            wait = self._last + gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def _store(self, key: str, reply: str) -> None:
        self.cache[key] = reply
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "model": self.cfg.model, "reply": reply}, ensure_ascii=False) + "\n")
