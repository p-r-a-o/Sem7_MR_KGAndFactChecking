"""Node addition: fetch facts the local graph is missing from the database it was built from (DBpedia).

Used by claim verification when a claim entity is not in the graph, or when claim entities cannot be
connected. Retrieved facts go into the graph's *overlay*: new entities become new nodes, every new
edge records its provenance (endpoint + retrieval date) and is cited as such in the proof, and the
mined rule statistics stay those of the frozen graph. Every addition is also appended to
artifacts/expath/<dataset>/additions.jsonl so the added information can be audited or merged later.

Only relations already in the graph's vocabulary are added (the rules and the relevance model know
nothing about others), and literal values use FactKG's '"value"' convention. Stdlib-only.
"""
import datetime as _dt
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from modules.expath.config import EnrichConfig
from modules.expath.graph import IndexedGraph

RESOURCE = "http://dbpedia.org/resource/"
ONTOLOGY = "http://dbpedia.org/ontology/"
_IRI_SAFE = "/:,()'!*&+;=@$-._~"

_QUERY_OUT = """SELECT ?p ?o WHERE {{ <{iri}> ?p ?o .
  FILTER(STRSTARTS(STR(?p), "{ont}") && !STRSTARTS(STR(?p), "{ont}wikiPage")) }} LIMIT {limit}"""
_QUERY_IN = """SELECT ?s ?p WHERE {{ ?s ?p <{iri}> .
  FILTER(STRSTARTS(STR(?p), "{ont}") && !STRSTARTS(STR(?p), "{ont}wikiPage")) }} LIMIT {limit}"""


def resource_iri(entity_id: str) -> str:
    return RESOURCE + urllib.parse.quote(entity_id, safe=_IRI_SAFE)


def _term(binding: dict) -> Optional[str]:
    """SPARQL JSON term -> graph entity ID (DBpedia resource name, or '"literal"')."""
    if binding["type"] == "uri":
        v = binding["value"]
        return urllib.parse.unquote(v[len(RESOURCE):]) if v.startswith(RESOURCE) else None
    return '"' + binding["value"] + '"'


def _select(endpoint: str, query: str, timeout: float) -> list[dict]:
    url = endpoint + "?" + urllib.parse.urlencode({"query": query, "format": "application/sparql-results+json"})
    req = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json",
                                               "User-Agent": "kg-reasoning-course-project/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["results"]["bindings"]


def fetch_facts(entity_id: str, cfg: EnrichConfig) -> list[tuple[str, str, str]]:
    """Forward (h, relation, t) facts touching the entity, DBpedia ontology relations only."""
    iri = resource_iri(entity_id)
    facts = []
    for b in _select(cfg.endpoint, _QUERY_OUT.format(iri=iri, ont=ONTOLOGY, limit=cfg.max_triples_per_entity),
                     cfg.timeout):
        o = _term(b["o"])
        if o is not None:
            facts.append((entity_id, b["p"]["value"][len(ONTOLOGY):], o))
    for b in _select(cfg.endpoint, _QUERY_IN.format(iri=iri, ont=ONTOLOGY, limit=cfg.max_triples_per_entity),
                     cfg.timeout):
        s = _term(b["s"])
        if s is not None and not s.startswith('"'):
            facts.append((s, b["p"]["value"][len(ONTOLOGY):], entity_id))
    return facts


def label_of(entity_id: str) -> str:
    return entity_id[1:-1] if entity_id.startswith('"') and entity_id.endswith('"') else entity_id.replace("_", " ")


def enrich(g: IndexedGraph, entity_ids: list[str], cfg: EnrichConfig, log_path: Optional[Path] = None,
           done: Optional[set[str]] = None) -> dict:
    """Retrieve and add facts for the given entities. Returns a summary for the explanation text.
    `done` (mutated) remembers entities already fetched so each is retrieved once per process."""
    done = done if done is not None else set()
    today = _dt.date.today().isoformat()
    source = f"DBpedia SPARQL {cfg.endpoint}, retrieved {today}"
    summary = {"entities_queried": [], "new_nodes": [], "facts_added": 0, "skipped_relations": 0, "errors": []}
    added_rows = []
    for e in entity_ids:
        if e in done:
            continue
        done.add(e)
        summary["entities_queried"].append(e)
        try:
            facts = fetch_facts(e, cfg)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as err:
            summary["errors"].append(f"{e}: {err}")
            continue
        for h, r, t in facts:
            rel = g.rel2idx.get(r)
            if rel is None:
                summary["skipped_relations"] += 1
                continue
            new = [x for x in (h, t) if x not in g.ent2idx]
            hi, ti = g.add_entity(h, label_of(h)), g.add_entity(t, label_of(t))
            if g.add_fact(hi, rel, ti, source):
                summary["facts_added"] += 1
                summary["new_nodes"].extend(new)
                added_rows.append({"triple": [h, r, t], "source": cfg.endpoint, "retrieved": today,
                                   "new_nodes": new})
    if log_path is not None and added_rows:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            for row in added_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["new_nodes"] = list(dict.fromkeys(summary["new_nodes"]))
    return summary
