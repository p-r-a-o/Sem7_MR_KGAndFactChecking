"""Readable relation names for display. Stdlib-only.

Knowledge-graph relation IDs are long ("/award/award_winner/awards_won./award/award_honor/award_winner")
or camelCase ("birthPlace"). `short_names` maps every relation of a graph to a short phrase
("award winner", "birth place"); when two relations would get the same phrase, the subject type is
prefixed ("award winning work: award winner") so the phrase stays unambiguous within that graph.
Display only: proofs and rules keep the full IDs so they can be checked.
"""
import re
from typing import Iterable

_CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
INVERSE_SUFFIX = "^-1"


def _words(segment: str) -> str:
    seg = segment.replace("_", " ").replace("-", " ").strip()
    if " " not in seg and any(c.isupper() for c in seg[1:]):         # camelCase -> words
        seg = " ".join(p.lower() for p in _CAMEL.findall(seg)) or seg
    return seg


def _candidates(rel: str) -> list[str]:
    """Phrases for one relation, shortest first; the last one is the full ID."""
    if "/" not in rel:
        return [_words(rel.rsplit(":", 1)[-1]), rel]
    parts = [p.strip("/").split("/") for p in rel.split(".")]
    role = _words(parts[-1][-1])
    subject = _words(parts[0][-2]) if len(parts[0]) >= 2 else ""
    via = _words(parts[0][-1]) if len(parts) > 1 else ""
    out = [role]
    if subject and subject == role and via:                          # 'award winner: award winner' says nothing
        out.append(f"{role} ({via})")
    elif subject:
        out.append(f"{subject}: {role}")
    if subject and via:
        out.append(f"{subject} ({via}): {role}")
    if subject and len(parts[0]) >= 3:                               # same subject type in two domains
        out.append(f"{_words(parts[0][-3])} {subject} ({via}): {role}" if via else f"{_words(parts[0][-3])} {subject}: {role}")
    if len(parts) > 1 and len(parts[-1]) >= 3:                       # ... and differing only in the record type
        out.append(f"{_words(parts[0][-3])} {subject}: {role} ({_words(parts[-1][-3])} {_words(parts[-1][-2])})")
    return out + [rel]


def short_names(relations: Iterable[str]) -> dict[str, str]:
    """relation ID -> shortest phrase that no other relation of the same graph shares."""
    rels = list(dict.fromkeys(relations))
    cands = {r: _candidates(r) for r in rels}
    level = {r: 0 for r in rels}
    while True:
        seen: dict[str, list[str]] = {}
        for r in rels:
            seen.setdefault(cands[r][level[r]], []).append(r)
        clashes = [r for group in seen.values() if len(group) > 1 for r in group
                   if level[r] < len(cands[r]) - 1]
        if not clashes:
            return {r: cands[r][level[r]] for r in rels}
        for r in clashes:
            level[r] += 1


def display(rel: str, names: dict[str, str]) -> str:
    """Short phrase for a relation, also for the shared inverse convention ('name^-1')."""
    if rel.endswith(INVERSE_SUFFIX):
        base = rel[: -len(INVERSE_SUFFIX)]
        return f"{names.get(base, base)} (inverse)"
    return names.get(rel, rel)
