"""The base link-prediction model that eXpath explains, scored with numpy only.

eXpath explains predictions of an embedding model (ComplEx / ConvE / TransE in its repo). Training
needs torch + a GPU and runs in the reference repo (scripts/train_kge.py); the serving environment
only reads the exported embeddings, so it stays numpy-only (torch is not available everywhere we run).

Export layout (artifacts/expath/<dataset>/kge/):
    entity_emb.npy    float32 [num_entities, 2d]   real | imaginary halves (eXpath ComplEx layout)
    relation_emb.npy  float32 [2 * num_relations, 2d]   inverse relation of r is row r + num_relations
    meta.json         {"model": "complex", "entities": [...], "relations": [...]}  names as eXpath read them
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np

from modules.expath.graph import IndexedGraph


def expath_name(name: str) -> str:
    """eXpath's dataset reader lower-cases names and strips ',', ':' and ';' (dataset.read_txt)."""
    return name.lower().replace(",", "").replace(":", "").replace(";", "")


class ComplExScorer:
    def __init__(self, ent: np.ndarray, rel: np.ndarray, ent_rows: np.ndarray, rel_rows: np.ndarray):
        self.ent, self.rel = ent, rel
        self.ent_rows = ent_rows      # graph entity index -> embedding row (-1 if unknown to the KGE)
        self.rel_rows = rel_rows      # graph doubled relation index -> embedding row (-1 if unknown)
        self.d = d = ent.shape[1] // 2
        # Tail embeddings re-ordered into graph order once, real and imaginary halves contiguous, so
        # scoring is two mat-vec products with no per-call copy (unknown entities get zero rows, masked).
        self.known = ent_rows >= 0
        tails = np.zeros((len(ent_rows), 2 * d), dtype=np.float32)
        tails[self.known] = ent[ent_rows[self.known]]
        self.t_re = np.ascontiguousarray(tails[:, :d])
        self.t_im = np.ascontiguousarray(tails[:, d:])

    @classmethod
    def load(cls, folder: Path, g: IndexedGraph) -> Optional["ComplExScorer"]:
        if not (folder / "meta.json").exists():
            return None
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        if meta.get("model") != "complex":
            raise ValueError(f"Only ComplEx exports are supported, got {meta.get('model')!r}")
        ent = np.load(folder / "entity_emb.npy").astype(np.float32)
        rel = np.load(folder / "relation_emb.npy").astype(np.float32)
        e_idx = {n: i for i, n in enumerate(meta["entities"])}
        r_idx = {n: i for i, n in enumerate(meta["relations"])}
        R_kge = len(meta["relations"])
        ent_rows = np.array([e_idx.get(expath_name(e), -1) for e in g.ent_ids], dtype=np.int64)
        fwd = [r_idx.get(expath_name(r), -1) for r in g.rel_names]
        rel_rows = np.array(fwd + [x + R_kge if x >= 0 else -1 for x in fwd], dtype=np.int64)
        return cls(ent, rel, ent_rows, rel_rows)

    def score_all(self, src: int, rel: int) -> np.ndarray:
        """Score of (src, rel, e) for every graph entity e; -inf where the KGE has no embedding.
        Same formula as eXpath ComplEx.score_embeddings: Re(<h, r, conj(t)>)."""
        hs, rs = self.ent_rows[src] if src < len(self.ent_rows) else -1, self.rel_rows[rel]
        if hs < 0 or rs < 0:
            return np.full(len(self.ent_rows), -np.inf, dtype=np.float32)
        d = self.d
        h, r = self.ent[hs], self.rel[rs]
        q_re = h[:d] * r[:d] - h[d:] * r[d:]
        q_im = h[:d] * r[d:] + h[d:] * r[:d]
        out = self.t_re @ q_re + self.t_im @ q_im
        out[~self.known] = -np.inf
        return out
