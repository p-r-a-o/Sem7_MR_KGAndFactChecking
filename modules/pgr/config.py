"""PGR module configuration. Paper settings (Hao & Wu, Findings of EMNLP 2025) are marked as such;
the rest exists because we run on a free LLM API tier."""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from common.paths import artifacts_root

GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


@dataclass
class LLMConfig:
    # Any OpenAI-compatible chat-completions endpoint works; default is Gemini's.
    url: str = field(default_factory=lambda: os.environ.get("PGR_LLM_URL", GEMINI_OPENAI_URL))
    model: str = field(default_factory=lambda: os.environ.get("PGR_MODEL", "gemini-2.5-flash"))
    api_key_env: str = "GEMINI_API_KEY"          # the key is read from this environment variable, never stored
    temperature: float = 0.2                     # paper
    top_p: float = 0.1                           # paper
    max_tokens: int = 4096                       # paper: 1024; Gemini "thinking" models spend output tokens
                                                 # on reasoning first, so 1024 can leave an empty answer
    requests_per_minute: float = field(default_factory=lambda: float(os.environ.get("PGR_RPM", "4")))
                                                 # stay under the free tier's per-minute cap (AI Studio, RPM
                                                 # column): 5 for the Gemini Flash models, 10-15 for Flash Lite
    reasoning_effort: str = field(default_factory=lambda: os.environ.get("PGR_REASONING", ""))
                                                 # optional OpenAI-style "reasoning_effort" (e.g. "low") sent to
                                                 # the endpoint; empty = not sent (the model's default)
    timeout: float = 120.0
    retries: int = 3                             # paper: 3 attempts per call
    max_prompt_chars: int = 8192                 # paper: longer MATCH/SEARCH prompts are skipped (no LLM call)


@dataclass
class PGRConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    artifact_root: Path = field(default_factory=artifacts_root)
    max_search_results: int = 50                 # entities kept from one SEARCH (ours; bounds MATCH fan-out)
    evidence_shown: int = 12                     # matched triples listed in an explanation

    def eval_path(self, split: str) -> Path:
        """Evaluation results, one file per model (and reasoning setting): results of different
        models must never be pooled into one score."""
        tag = re.sub(r"[^A-Za-z0-9._-]+", "_", self.llm.model) + (f"_{self.llm.reasoning_effort}"
                                                                  if self.llm.reasoning_effort else "")
        return self.artifact_root / "pgr" / f"eval_{split}_{tag}.jsonl"

    def cache_path(self) -> Path:
        """Every LLM reply is cached here, keyed by model + prompt, so reruns cost no quota."""
        return self.artifact_root / "pgr" / "llm_cache.jsonl"
