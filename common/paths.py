"""Repo-relative locations. Override with env vars if data/artifacts live elsewhere."""
import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return Path(os.environ.get("KGX_DATA_DIR", repo_root() / "data"))


def artifacts_root() -> Path:
    return Path(os.environ.get("KGX_ARTIFACTS_DIR", repo_root() / "artifacts"))


def claims_dir(dataset: str) -> Path:
    return data_root() / "claims" / dataset
