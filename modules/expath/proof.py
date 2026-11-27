"""Proofs: evidence -> shared `Explanation`, and an independent checker for those proofs.

A proof here is *checkable* in two ways:
  1. every step is a fact (h, r, t) that can be looked up in the graph (or, for facts added at run
     time, at the cited source), and the steps chain from the query entity to the answer;
  2. the rule in `Explanation.rule` comes with SC / HC / support that anyone can recompute from the
     graph (`check_explanation` does exactly that).
"""
import re
from typing import Optional

from common.schema import Explanation, ExplanationStep
from modules.expath.graph import Hop, IndexedGraph
from modules.expath.reasoner import Firing, TripleEvidence
from modules.expath.rules import (Rule, cp_metrics, formal_rule, metrics_readable, metrics_text,
                                  rule_readable, uc_metrics)

PROOF_TAG = "proof"          # prefix of step notes that belong to the proof chain
NL, PARA = "\n", "\n\n"      # explanation text is Markdown


def path_text(g: IndexedGraph, path) -> str:
    """The facts of a grounded path, each written as stored: 'A —[r]→ B, C —[s]→ B'."""
    return ", ".join(g.hop_text(hop) for hop in path)


def hop_step(g: IndexedGraph, hop: Hop, note: str) -> ExplanationStep:
    """A step: the fact (full IDs, checkable) plus a short note. The UI prints the fact itself."""
    return ExplanationStep(triple=g.fact(hop), note=f"{note} · source: {g.provenance(hop)}")


def firing_line(g: IndexedGraph, f: Firing) -> str:
    example = f": {path_text(g, f.groundings[0])}" if f.groundings else ""
    return f"- {rule_readable(g, f.rule)}  ({f.rule.sc:.0%} right){example}"


def firings_explanation(g: IndexedGraph, intro: str, firings: list[Firing], confidence: float,
                        critical: Optional[list[tuple[Hop, float]]] = None, rules_shown: int = 3,
                        facts_shown: int = 3, extra_steps: Optional[list[ExplanationStep]] = None,
                        n_total: Optional[int] = None) -> Explanation:
    """Best firing = the proof (rule + its grounding as steps); a few more agreeing rules are listed.
    Text is Markdown for people; `rule` holds the formal rule + metrics that `check_explanation` re-verifies."""
    steps = list(extra_steps or [])
    if not firings:
        return Explanation(kind="path", steps=steps, confidence=confidence,
                           text=f"{intro}{PARA}No rule passing eXpath's quality thresholds connects these entities.")
    best = firings[0]
    n = len(best.groundings[0])
    for i, hop in enumerate(best.groundings[0], 1):
        steps.append(hop_step(g, hop, f"{PROOF_TAG}: fact {i} of {n} that the rule needs"))
    shown = {s.triple for s in steps}
    for hop, score in (critical or [])[:facts_shown]:
        if g.fact(hop) not in shown and score >= 0.01:
            steps.append(hop_step(g, hop, f"also important (eXpath fact score {score:.2f})"))
            shown.add(g.fact(hop))
    text = (f"{intro}{PARA}"
            f"**Rule:** {rule_readable(g, best.rule)}{NL}"
            f"- {metrics_readable(g, best.rule)};{NL}"
            f"- it applies here because the graph contains **{path_text(g, best.groundings[0])}**.")
    total = n_total if n_total is not None else len(firings)
    others = [firing_line(g, f) for f in firings[1:rules_shown]]
    if others:
        plural = "s" if total > 2 else ""
        text += f"{PARA}**{total - 1} more rule{plural} agree** (combined confidence {confidence:.2f}); strongest:{NL}"
        text += NL.join(others)
    return Explanation(kind="path", steps=steps, text=text, rule=formal_rule(g, best.rule), confidence=confidence)


def triple_explanation(g: IndexedGraph, intro: str, ev: TripleEvidence, rules_shown: int = 3,
                       facts_shown: int = 3) -> Explanation:
    ex = firings_explanation(g, intro, ev.firings, ev.confidence, ev.critical_facts, rules_shown, facts_shown)
    ex.text += f"{PARA}<sub>eXpath searched {ev.n_paths} connecting paths of up to 3 facts.</sub>"
    return ex


# ----------------------------------------------------------------------------- checking

_RULE_RE = re.compile(r"^\s*(?P<head>[^(]+)\((?P<hargs>[^)]*)\)\s*<-\s*(?P<body>.+)$")
_ARG = r'(?:"[^"]*"\s*\[[^\]]*\]|[^,()]+)'          # a variable, or a constant '"label" [id]'
_ATOM_RE = re.compile(r'([^(),\s][^(),]*)\((' + _ARG + r'),\s*(' + _ARG + r')\)')


def _parse_rule(g: IndexedGraph, text: str) -> Optional[Rule]:
    """Parse a rule string produced by `rule_text` back into a Rule with zeroed metrics."""
    m = _RULE_RE.match(text)
    if not m:
        return None
    head_part, _, body_part = text.partition("<-")
    atoms = _ATOM_RE.findall(head_part)[:1] + _ATOM_RE.findall(body_part)
    if len(atoms) < 2:
        return None

    def rel_of(name, a, b, x):  # relation id oriented from variable x
        r = g.rel2idx.get(name.strip())
        if r is None:
            return None
        return r if a.strip() == x else r + g.R

    (hn, ha, hb), body = atoms[0], atoms[1:]
    if hb.strip().startswith('"') or ha.strip().startswith('"'):       # Uc: head(X, "c") <- b(X, "c'")
        def const(s):  # '"label" [id]' -> entity index
            m_id = re.search(r"\[(.+)\]\s*$", s.strip())
            return g.ent2idx.get(m_id.group(1)) if m_id else None
        x_first = not ha.strip().startswith('"')
        head = rel_of(hn, ha, hb, "X")
        target = const(hb if x_first else ha)
        bn, ba, bb = body[0]
        b = rel_of(bn, ba, bb, "X")
        c2 = const(bb if not ba.strip().startswith('"') else ba)
        if None in (head, target, b, c2):
            return None
        return Rule("Uc", head, (b,), 0, 0, 0, target=target, body_const=c2)
    head = rel_of(hn, ha, hb, "X")
    rels, var = [], "X"
    for bn, ba, bb in body:
        r = rel_of(bn, ba, bb, var)
        if r is None:
            return None
        rels.append(r)
        var = bb.strip() if ba.strip() == var else ba.strip()
    return None if head is None else Rule("CP", head, tuple(rels), 0, 0, 0)


def check_explanation(g: IndexedGraph, ex: Explanation) -> dict:
    """Re-verify a proof against the graph: every cited fact exists, and the cited rule's metrics
    recompute to the stated values. Returns a report dict (all_facts_ok, rule_ok, details)."""
    facts = []
    for s in ex.steps:
        if s.triple is None:
            continue
        h, r, t = s.triple
        hi, ri, ti = g.ent2idx.get(h), g.rel2idx.get(r), g.ent2idx.get(t)
        ok = None not in (hi, ri, ti) and g.has_edge(hi, ri, ti)
        facts.append({"triple": list(s.triple), "in_graph": ok, "external": "source: knowledge graph" not in s.note})
    report = {"facts": facts, "all_facts_ok": all(f["in_graph"] for f in facts)}
    if ex.rule:
        rule_str, _, stated_metrics = ex.rule.partition("  % ")
        rule = _parse_rule(g, rule_str)
        if rule is None:
            report["rule_ok"] = False
            report["rule_error"] = "could not parse rule"
        else:
            if rule.kind == "CP":
                supp, n_body, n_head = cp_metrics(g, rule.head, rule.body)
            else:
                supp, n_body, n_head = uc_metrics(g, rule.head, rule.target, rule.body[0], rule.body_const)
            recomputed = Rule(rule.kind, rule.head, rule.body, supp, n_body, n_head, rule.target, rule.body_const)
            stated = metrics_text(recomputed)
            report["recomputed_metrics"] = stated
            report["rule_ok"] = stated == stated_metrics.strip() if stated_metrics else stated in ex.text
    return report
