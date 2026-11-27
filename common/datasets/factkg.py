"""FactKG: DBpedia 2015 subgraph around the FactKG claim entities (Kim et al., ACL 2023).

Built by scripts/prepare_factkg.py into data/factkg/ (graph triples in train.txt; valid/test are
empty because FactKG holds out claims, not triples) and data/claims/factkg/*.jsonl.
Entity IDs are DBpedia resource names ("Pleasant_Springs,_Wisconsin"); literals keep FactKG's quotes ('"94802.4"').
"""
from common.datasets.files import DatasetSpec

SPEC = DatasetSpec(name="factkg")
