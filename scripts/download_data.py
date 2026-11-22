"""Download FB15k-237 / WN18RR splits (+ entity names) into data/<dataset>/. Stdlib-only.

Splits come from the PowerLink repo (same files and IDs its pretrained models index, so the
shared loader and the PowerLink artifacts agree). Entity names come from the KG-BERT repo's
entity2text.txt; these URLs were NOT verified when this was written - if a label download fails
the script warns and the loader falls back to raw IDs (claims then read badly).

Usage: python scripts/download_data.py [--dataset fb15k237 wn18rr]
"""
import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_root  # noqa: E402

SPLIT_URL = "https://raw.githubusercontent.com/Jiangnan0522/power-link/HEAD/datasets/{ds}/{file}"
LABEL_URLS = {
    "fb15k237": [
        "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/FB15k-237/entity2text.txt",
        "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/FB15k/entity2text.txt",
    ],
    "wn18rr": [
        "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/WN18RR/entity2text.txt",
    ],
}


def fetch(url: str, dest: Path) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            dest.write_bytes(resp.read())
        print(f"  ok   {url}")
        return True
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"  FAIL {url} ({e})")
        return False


def download(dataset: str) -> None:
    folder = data_root() / dataset
    folder.mkdir(parents=True, exist_ok=True)
    print(f"[{dataset}] -> {folder}")
    for fname in ("train.txt", "valid.txt", "test.txt"):
        if not fetch(SPLIT_URL.format(ds=dataset, file=fname), folder / fname):
            raise SystemExit(f"Could not download {fname}; check your network.")
    if not any(fetch(u, folder / "entity2text.txt") for u in LABEL_URLS[dataset]):
        print(f"  WARNING: no entity names for {dataset}; place entity2text.txt ('<id>\\t<name>') in {folder}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["fb15k237", "wn18rr"], choices=["fb15k237", "wn18rr"])
    for ds in ap.parse_args().dataset:
        download(ds)
