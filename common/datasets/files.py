"""Readers shared by all datasets. Dataset quirks live in the per-dataset specs, not in modules."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from common.datasets.base import INVERSE_SUFFIX, KnowledgeGraph
from common.paths import data_root
from common.schema import Triple


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    label_file: str = "entity2text.txt"          # optional "<id>\t<name>" file inside data/<name>/
    clean_label: Callable[[str], str] = str.strip


def read_triples(path: Path) -> list[Triple]:
    """Read tab-separated (head, relation, tail) lines. Tolerates CRLF line endings."""
    triples: list[Triple] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def read_labels(path: Path, clean: Callable[[str], str]) -> dict[str, str]:
    """Read "<id>\t<text>" lines. Missing file -> empty dict (IDs are then used as labels)."""
    if not path.exists():
        return {}
    labels: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                labels[parts[0]] = clean(parts[1])
    return labels


def build_neighbors(triples: list[Triple]) -> dict[str, list[tuple[str, str]]]:
    neighbors: dict[str, list[tuple[str, str]]] = {}
    for h, r, t in triples:
        neighbors.setdefault(h, []).append((r, t))
        neighbors.setdefault(t, []).append((r + INVERSE_SUFFIX, h))
    return neighbors


def build_knowledge_graph(spec: DatasetSpec, root: Optional[Path] = None) -> KnowledgeGraph:
    folder = (root or data_root()) / spec.name
    splits = {s: read_triples(folder / f"{s}.txt") for s in ("train", "valid", "test")}
    labels = read_labels(folder / spec.label_file, spec.clean_label)
    entities = {x for ts in splits.values() for h, _, t in ts for x in (h, t)}
    return KnowledgeGraph(
        name=spec.name,
        train=splits["train"], valid=splits["valid"], test=splits["test"],
        entity_labels={e: labels.get(e, e) for e in sorted(entities)},
        relations=sorted({r for ts in splits.values() for _, r, _ in ts}),
        neighbors=build_neighbors(splits["train"]),
    )
