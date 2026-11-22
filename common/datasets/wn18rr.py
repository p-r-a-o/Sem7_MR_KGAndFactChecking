"""WN18RR: entities are WordNet synset offsets (e.g. 00260881)."""
import re

from common.datasets.files import DatasetSpec

_SYNSET_SUFFIX = re.compile(r"\.[nvasr]\.\d+$")


def clean_label(text: str) -> str:
    """'about_face.n.01' -> 'about face'. Plain lemma text passes through unchanged."""
    return _SYNSET_SUFFIX.sub("", text.strip()).replace("_", " ")


SPEC = DatasetSpec(name="wn18rr", clean_label=clean_label)
