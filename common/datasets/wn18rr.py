"""WN18RR: entities are WordNet synset offsets (e.g. 00260881)."""
import re

from common.datasets.files import DatasetSpec

_SYNSET_SUFFIX = re.compile(r"\.[nvasr]\.\d+$")


def clean_label(text: str) -> str:
    """Keep only the lemma, so proofs and claims stay readable.

    entity2text.txt stores "<lemma>, <dictionary definition>", e.g.
    "stool, solid excretory product evacuated from the bowels" -> "stool".
    Also handles synset-style labels: 'about_face.n.01' -> 'about face'.
    """
    lemma = text.strip().split(",")[0].strip()
    return _SYNSET_SUFFIX.sub("", lemma).replace("_", " ") or text.strip()


def gloss(text: str, max_words: int = 6) -> str:
    """The definition half of "<lemma>, <definition>", trimmed to something a sentence can carry.

    "response, a result" -> "a result". Cut at the first ';' (WordNet uses it before usage
    examples) and capped at `max_words`, because this ends up inside a generated claim.
    """
    parts = text.strip().split(",", 1)
    if len(parts) < 2:
        return ""
    definition = parts[1].split(";")[0].strip().strip('"')
    words = definition.split()
    return " ".join(words[:max_words]) if words else ""


SPEC = DatasetSpec(name="wn18rr", clean_label=clean_label, gloss=gloss)
