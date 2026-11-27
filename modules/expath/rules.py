"""eXpath's ontological rules, their quality metrics, and offline mining.

Two rule types from the eXpath paper / process.py are used:
  CP (closed path):   r(X, Y) <- b1(X, Z1), b2(Z1, Z2), ..., bk(Zk-1, Y)
  Uc (constant/"property transition"):  r(X, c) <- b(X, c')
eXpath's third type, Ud, is weighted only by the KGE relevance from Kelpie-style post-training, which
this module does not run (see README, "Deviations"), so it is left out.

Metrics follow process.py exactly:
  supp = #(x, y) with both body and head true,  #body = #(x, y) with body true (x != y),
  #head = #facts of r,  SC = supp / #body,  HC = supp / #head,
  rule score = SC * supp / (supp + minSupp)   (minSupp = 10).
`cp_metrics` is `calculate_rule_metrics_with_matrix` (binarised sparse products, diagonal removed),
computed in row blocks so a dense intermediate never has to fit in memory.
`uc_metrics` is `calculate_Uc_metrics`.
"""
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from scipy import sparse

from modules.expath.config import MiningConfig, RuleQuality
from modules.expath.graph import IndexedGraph, enumerate_paths, relation_path


@dataclass(frozen=True)
class Rule:
    kind: str                         # "CP" | "Uc"
    head: int                         # head relation, doubled space
    body: tuple[int, ...]             # CP: relation path; Uc: (body relation,)
    supp: int
    n_body: int
    n_head: int
    target: Optional[int] = None      # Uc: constant c in r(X, c)
    body_const: Optional[int] = None  # Uc: constant c' in b(X, c')

    @property
    def sc(self) -> float:
        return self.supp / self.n_body if self.n_body else 0.0

    @property
    def hc(self) -> float:
        return self.supp / self.n_head if self.n_head else 0.0

    def score(self, smoothing: int = 10) -> float:
        """eXpath: SC * supp / (supp + minSupp)."""
        return self.sc * self.supp / (self.supp + smoothing) if self.supp else 0.0

    def passes(self, q: RuleQuality) -> bool:
        return self.sc >= q.min_sc and self.hc >= q.min_hc and self.supp > 0

    def inverse(self, R: int) -> "Rule":
        """r(X,Y) <- b1..bk  is the same rule as  r^-1(Y,X) <- bk^-1..b1^-1 (identical metrics)."""
        if self.kind != "CP":
            raise ValueError("only CP rules have a direct inverse")
        inv = lambda r: r + R if r < R else r - R  # noqa: E731
        return Rule("CP", inv(self.head), tuple(inv(r) for r in reversed(self.body)),
                    self.supp, self.n_body, self.n_head)


# ----------------------------------------------------------------------------- rendering

def _atom(g: IndexedGraph, rel: int, a: str, b: str) -> str:
    return f"{g.rel_base(rel)}({a}, {b})" if rel < g.R else f"{g.rel_base(rel)}({b}, {a})"


def constant_text(g: IndexedGraph, idx: int) -> str:
    """Rule constant: readable label plus the canonical ID, e.g. '"Male" [/m/05zppz]'.
    Parentheses and commas are dropped from the label so the rule string stays parseable."""
    label = g.label(idx).replace("(", "").replace(")", "").replace(",", "").replace('"', "")
    return f'"{label}" [{g.entity_id(idx)}]'


def rule_text(g: IndexedGraph, rule: Rule) -> str:
    """Datalog-style rule string, e.g. 'nationality(X, Y) <- place_of_birth(X, A), country(A, Y)'."""
    if rule.kind == "Uc":
        head = _atom(g, rule.head, "X", constant_text(g, rule.target))
        body = _atom(g, rule.body[0], "X", constant_text(g, rule.body_const))
        return f"{head} <- {body}"
    names = ["X"] + [chr(ord("A") + i) for i in range(len(rule.body) - 1)] + ["Y"]
    body = ", ".join(_atom(g, r, names[i], names[i + 1]) for i, r in enumerate(rule.body))
    return f"{_atom(g, rule.head, 'X', 'Y')} <- {body}"


def _fact(g: IndexedGraph, rel: int, a: str, b: str) -> str:
    """'a —[name]→ b', written in the stored direction of the relation."""
    if rel >= g.R:
        a, b = b, a
    return f"{a} —[{g.rel_short(rel)}]→ {b}"


def rule_readable(g: IndexedGraph, rule: Rule) -> str:
    """The rule as a sentence: 'if Y —[award nominee]→ X, then X —[award winner]→ Y'."""
    if rule.kind == "Uc":
        body = _fact(g, rule.body[0], "X", f'"{g.label(rule.body_const)}"')
        head = _fact(g, rule.head, "X", f'"{g.label(rule.target)}"')
        return f"if {body}, then {head}"
    names = ["X"] + [chr(ord("A") + i) for i in range(len(rule.body) - 1)] + ["Y"]
    body = " and ".join(_fact(g, r, names[i], names[i + 1]) for i, r in enumerate(rule.body))
    return f"if {body}, then {_fact(g, rule.head, 'X', 'Y')}"


def metrics_readable(g: IndexedGraph, rule: Rule) -> str:
    """SC and HC in words (same numbers as `metrics_text`)."""
    if rule.kind == "Uc":      # head count of a Uc rule = facts with that constant, e.g. born in Mumbai
        known = f'known "{_fact(g, rule.head, "?", g.label(rule.target))}" facts'
    else:
        known = f"known [{g.rel_short(rule.head)}] facts"
    return (f"right in {rule.sc:.0%} of the cases where its *if* part holds ({rule.supp:,} of {rule.n_body:,}); "
            f"explains {rule.hc:.0%} of all {known} ({rule.supp:,} of {rule.n_head:,})")


def formal_rule(g: IndexedGraph, rule: Rule) -> str:
    """Machine-checkable form: Datalog rule with full relation IDs, then '%' and its stated metrics."""
    return f"{rule_text(g, rule)}  % {metrics_text(rule)}"


def metrics_text(rule: Rule) -> str:
    return (f"SC={rule.sc:.3f} (supp {rule.supp} / body {rule.n_body}), "
            f"HC={rule.hc:.3f} (supp {rule.supp} / head {rule.n_head})")


# ----------------------------------------------------------------------------- metrics

def path_instances(g: IndexedGraph, body: Sequence[int]) -> int:
    """Number of (not de-duplicated) groundings of a relation path: 1^T A1 A2 ... Ak 1.
    O(nnz) via sparse mat-vec products; an upper bound on the work `cp_metrics` would do."""
    x = np.ones(g.N, dtype=np.float64)
    for r in reversed(body):
        x = g.matrix(r) @ x
    return int(x.sum())


def cp_metrics(g: IndexedGraph, head: int, body: Sequence[int], row_chunk: int = 4096) -> tuple[int, int, int]:
    """(supp, #body, #head) of CP rule head <- body on the base graph (eXpath's matrix method)."""
    head_m = g.matrix(head)
    mats = [g.matrix(r) for r in body]
    rows_with_body = np.flatnonzero(np.diff(mats[0].indptr))
    supp = n_body = 0
    for lo in range(0, len(rows_with_body), row_chunk):
        rows = rows_with_body[lo: lo + row_chunk]
        m = mats[0][rows]
        for nxt in mats[1:]:
            m = m @ nxt
            m.data[:] = 1                           # binarise, as eXpath does
            if not m.nnz:
                break
        m = m.tocoo()
        keep = rows[m.row] != m.col                 # eXpath: setdiag(0), X != Y
        r_idx, c_idx = rows[m.row[keep]], m.col[keep]
        n_body += int(keep.sum())
        if len(r_idx):
            supp += int(np.asarray(head_m[r_idx, c_idx]).ravel().sum())
    return supp, n_body, int(head_m.nnz)


def uc_metrics(g: IndexedGraph, head: int, target: int, body: int, body_const: int,
               xs_head: Optional[set] = None) -> tuple[int, int, int]:
    """(supp, #body, #head) of Uc rule head(X, target) <- body(X, body_const) (eXpath calculate_Uc_metrics).
    `xs_head` (entities X with head(X, target)) may be passed in when scoring many bodies for one head."""
    if xs_head is None:
        xs_head = set(g.neighbors_by_rel(target, g.inv(head)).tolist())   # X with head(X, target)
    xs_body = set(g.neighbors_by_rel(body_const, g.inv(body)).tolist())   # X with body(X, body_const)
    return len(xs_head & xs_body), len(xs_body), len(xs_head)


# ----------------------------------------------------------------------------- mining

def mine_cp_rules(g: IndexedGraph, cfg: MiningConfig, quality: RuleQuality,
                  relations: Optional[Iterable[int]] = None,
                  progress: Callable[[str], None] = print) -> list[Rule]:
    """eXpath stages 1-2 run once per relation instead of once per explained prediction:
    aggregate relation paths between the endpoints of sampled facts of r, keep frequent ones,
    score them with `cp_metrics`, keep those passing eXpath's SC/HC filters. Inverse rules are added
    for free (same metrics), so head queries can apply them too."""
    rng = np.random.default_rng(cfg.seed)
    rules: list[Rule] = []
    rel_list = list(relations) if relations is not None else list(range(g.R))
    for n, rel in enumerate(rel_list):
        facts = np.flatnonzero(g.triples[:, 1] == rel)
        if not len(facts):
            continue
        sample = rng.choice(facts, size=min(cfg.samples_per_relation, len(facts)), replace=False)
        counts: Counter = Counter()
        for i in sample.tolist():
            h, _, t = g.triples[i].tolist()
            paths = enumerate_paths(g, h, t, cfg.max_path_len, exclude=(h, rel, t), max_paths=5000,
                                    hub_degree=cfg.hub_degree, max_start=cfg.max_start_neighbors, rng=rng)
            counts.update({relation_path(p) for p in paths})
        bodies = [b for b, c in counts.most_common() if c >= cfg.min_body_count and b != (rel,)]
        kept = 0
        for body in bodies[: cfg.max_bodies_per_relation]:
            supp, n_body, n_head = cp_metrics(g, rel, body, cfg.row_chunk)
            rule = Rule("CP", rel, body, supp, n_body, n_head)
            if rule.passes(quality):
                rules.extend((rule, rule.inverse(g.R)))
                kept += 1
        progress(f"[CP {n + 1}/{len(rel_list)}] {g.rel_names[rel]}: {len(bodies)} frequent paths, kept {kept}")
    return rules


def mine_uc_rules(g: IndexedGraph, cfg: MiningConfig, quality: RuleQuality,
                  progress: Callable[[str], None] = print) -> list[Rule]:
    """Uc rules head(X, c) <- b(X, c') for every (head, c) with >= uc_min_head facts, both directions.

    One sparse product per head relation: rows = constants c, columns = entities X with head(X, c);
    times the entity x feature matrix (feature = edge (b, c') leaving X) gives every supp at once.
    """
    R2 = 2 * g.R
    feat_key = g.rel * g.N + g.dst                       # feature id of edge X --b--> c'
    feat_ids, feat_col = np.unique(feat_key, return_inverse=True)
    F = sparse.csr_matrix((np.ones(len(feat_col), dtype=np.int32), (g.src, feat_col)),
                          shape=(g.N, len(feat_ids)))
    F.data[:] = 1
    feat_count = np.asarray(F.sum(axis=0)).ravel()        # #body for each feature
    rules: list[Rule] = []
    for head in range(R2):
        # X --head--> c  <=>  c --inv(head)--> X ; group by c
        m = g.matrix(g.inv(head))                           # rows c, cols X
        n_head = np.diff(m.indptr)
        consts = np.flatnonzero(n_head >= cfg.uc_min_head)
        if not len(consts):
            continue
        S = (m[consts] @ F).tocsr()                         # supp per (c, feature)
        own = head * g.N + consts                           # the feature head(X, c) itself
        kept = 0
        for i, c in enumerate(consts.tolist()):
            lo, hi = S.indptr[i], S.indptr[i + 1]
            cols, supp = S.indices[lo:hi], S.data[lo:hi]
            ok = supp >= cfg.uc_min_supp
            cols, supp = cols[ok], supp[ok]
            sc = supp / feat_count[cols]
            ok = (sc >= max(cfg.uc_min_sc, quality.min_sc)) & (supp / n_head[c] >= quality.min_hc) & \
                 (feat_ids[cols] != own[i])
            cols, supp = cols[ok], supp[ok]
            best = np.argsort(-(supp / feat_count[cols]) * supp / (supp + quality.supp_smoothing))
            for j in best[: cfg.uc_max_rules_per_head].tolist():
                key = int(feat_ids[cols[j]])
                b, c2 = divmod(key, g.N)
                if b == head and c2 == c:
                    continue
                rules.append(Rule("Uc", head, (int(b),), int(supp[j]), int(feat_count[cols[j]]),
                                  int(n_head[c]), target=int(c), body_const=int(c2)))
                kept += 1
        progress(f"[Uc {head + 1}/{R2}] {g.rel_name(head)}: {len(consts)} constants, kept {kept}")
    return rules


# ----------------------------------------------------------------------------- IO (by name, not index)

def save_rules(g: IndexedGraph, rules: Iterable[Rule], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rules:
            d = asdict(r)
            d["head"] = g.rel_name(r.head)
            d["body"] = [g.rel_name(b) for b in r.body]
            d["target"] = None if r.target is None else g.entity_id(r.target)
            d["body_const"] = None if r.body_const is None else g.entity_id(r.body_const)
            d["sc"], d["hc"] = round(r.sc, 6), round(r.hc, 6)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_rules(g: IndexedGraph, path: Path) -> list[Rule]:
    """Rules referring to relations/entities unknown to this graph are skipped."""
    rules = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            head = g.rel_index(d["head"])
            body = [g.rel_index(b) for b in d["body"]]
            if head is None or None in body:
                continue
            target = g.ent2idx.get(d["target"]) if d.get("target") is not None else None
            const = g.ent2idx.get(d["body_const"]) if d.get("body_const") is not None else None
            if d["kind"] == "Uc" and (target is None or const is None):
                continue
            rules.append(Rule(d["kind"], head, tuple(body), d["supp"], d["n_body"], d["n_head"], target, const))
    return rules
