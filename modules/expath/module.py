"""ExpathModule: eXpath (path-based, ontological closed-path rules) behind the shared ReasoningModule interface.

Link prediction
  * "kge" mode (when an exported base model is in artifacts/expath/<ds>/kge/): the KGE ranks all
    entities, and eXpath explains each top candidate (explain_triple). With `require_proof`
    (default) a candidate eXpath cannot prove is dropped: a score alone is not an answer.
  * "rules" mode (no KGE artifacts): the rules eXpath's metrics accept are applied forward from the
    query entity; candidates are ranked by the noisy-OR of their rules' scores.
  Head queries (?, r, t) are answered as (t, r^-1, ?); mined rules include their inverses.

Claim verification
  * claim_mode "triple" (FB15k-237, WN18RR): shared claim parser -> (h, r, t) -> stored fact, or
    eXpath proof above `support`, or a conflict (functional relation / better-proven alternative),
    else NOT_ENOUGH_INFO.
  * claim_mode "multi_entity" (FactKG): see factcheck.py. Optional node addition from DBpedia.

Every LinkPrediction / CVResult carries an Explanation whose steps are graph facts (with
provenance) and whose rule metrics can be re-checked with scripts/check_proof.py.
"""
from dataclasses import dataclass
import re
from typing import Optional

import numpy as np

from common.client import parse_claim
from common.datasets.base import KnowledgeGraph
from common.errors import ModuleError
from common.paths import claims_dir
from common.schema import CVResult, Explanation, ExplanationStep, LinkPrediction, LPResult
from common.services import get_service
from modules.expath.claims import EntityLinker, Lexicon, analyse, is_negated, load_entity_sets
from modules.expath.config import ExpathConfig
from modules.expath.enrich import enrich
from modules.expath.factcheck import decide, gather_evidence
from modules.expath.graph import IndexedGraph
from modules.expath.kge import ComplExScorer
from modules.expath.proof import NL, PROOF_TAG, firings_explanation, hop_step, triple_explanation
from modules.expath.reasoner import (RuleBase, apply_rules, explain_triple, ground_firings,
                                     rank_candidates)
from modules.expath.rules import load_rules


def functional_ratio(g: IndexedGraph) -> np.ndarray:
    """Per doubled relation: share of sources that have exactly one target in the train graph."""
    out = np.zeros(2 * g.R)
    if not len(g.triples):
        return out
    for col_src, offset in ((0, 0), (2, g.R)):
        rel_src, counts = np.unique(g.triples[:, [1, col_src]], axis=0, return_counts=True)
        n_src = np.bincount(rel_src[:, 0], minlength=g.R)
        n_single = np.bincount(rel_src[:, 0], weights=(counts == 1).astype(float), minlength=g.R)
        out[offset: offset + g.R] = np.divide(n_single, n_src, out=np.zeros(g.R), where=n_src > 0)
    return out


@dataclass
class _State:
    kg: KnowledgeGraph
    g: IndexedGraph
    rb: RuleBase
    kge: Optional[ComplExScorer]
    functional: np.ndarray
    claim_mode: str
    lexicon: Optional[Lexicon] = None
    entity_sets: Optional[dict] = None
    linker: Optional[EntityLinker] = None


class ExpathModule:
    name = "eXpath"

    def __init__(self, config: Optional[ExpathConfig] = None):
        self.cfg = config or ExpathConfig()
        self._st: Optional[_State] = None
        self._enriched: set[str] = set()

    # ------------------------------------------------------------------ loading

    def load(self, kg: KnowledgeGraph) -> None:
        g = IndexedGraph.from_kg(kg)
        art = self.cfg.artifacts(kg.name)
        rules = load_rules(g, art / "rules.jsonl") if (art / "rules.jsonl").exists() else []
        rb = RuleBase(rules, self.cfg.quality.supp_smoothing)
        kge = ComplExScorer.load(art / "kge", g) if self.cfg.use_kge else None
        mode = self.cfg.claim_mode.get(kg.name, self.cfg.default_claim_mode)
        st = _State(kg, g, rb, kge, functional_ratio(g), mode)
        if mode == "multi_entity":
            st.lexicon = Lexicon.load(art / "lexicon.json")
            st.entity_sets = load_entity_sets(sorted(claims_dir(kg.name).glob("*.jsonl")))
        self._st, self._enriched = st, set()
        print(f"[eXpath] {kg.name}: {g.N} entities, {len(g.triples)} train facts, {len(rules)} mined rules, "
              f"KGE {'loaded' if kge else 'absent (rules mode)'}, claim mode {mode}"
              + (f", lexicon {'loaded' if st.lexicon else 'MISSING'}" if mode == "multi_entity" else ""))

    def _need(self) -> _State:
        if self._st is None:
            raise ModuleError("NOT_LOADED", "eXpath: call load() first.")
        return self._st

    def _ent(self, e: str) -> int:
        idx = self._need().g.ent2idx.get(e)
        if idx is None:
            raise ModuleError("UNKNOWN_ENTITY", f"Unknown entity: {e!r}")
        return idx

    def _rel(self, r: str) -> int:
        idx = self._need().g.rel2idx.get(r)
        if idx is None:
            raise ModuleError("UNKNOWN_ENTITY", f"Unknown relation: {r!r}")
        return idx

    # ------------------------------------------------------------------ link prediction

    def predict_link(self, head: Optional[str], relation: str, tail: Optional[str], k: int = 10) -> LPResult:
        st = self._need()
        g = st.g
        reverse = head is None
        src = self._ent(tail if reverse else head)
        rel = self._rel(relation) + (g.R if reverse else 0)
        exclude = {src}
        if self.cfg.exclude_known:
            exclude |= set(g.neighbors_by_rel(src, rel).tolist())
        name = g.rel_short(rel)
        q = (f"? —[{name}]→ {g.label(src)}" if reverse else f"{g.label(src)} —[{name}]→ ?")
        stored = sorted(exclude - {src}, key=g.label)
        can_predict = st.kge is not None or bool(st.rb.cp_by_head or st.rb.uc_index)
        no_model = (f"Link prediction is not set up for '{st.kg.name}': it has no mined rules and no trained "
                    f"embedding model. This project does link prediction on FB15k-237 (pick fb15k237) and uses "
                    f"{st.kg.name} for claim verification.")
        if not can_predict and not (stored and self.cfg.show_known):
            raise ModuleError("INTERNAL", no_model)
        preds = ((self._lp_kge(src, rel, exclude, k, q) if st.kge is not None else self._lp_rules(src, rel, exclude, k, q))
                 if can_predict else [])
        if stored and self.cfg.show_known:
            # Interactive use: list the answers the graph already states first, each proven by the stored
            # fact itself, then the predicted (missing) ones. Off for evaluation, which counts list positions.
            shown = stored[: self.cfg.known_shown]
            known_preds = []
            for x in shown:
                hop = (src, rel, x)
                text = f"✅ **Already in the knowledge graph** (not a prediction): **{g.hop_text(hop)}**."
                if x == shown[-1] and len(stored) > len(shown):
                    text += f"{NL}{NL}<sub>{len(stored) - len(shown)} more stored answers not listed.</sub>"
                if x == shown[0] and not can_predict:
                    text += f"{NL}{NL}ℹ️ Only stored answers are shown. {no_model}"
                known_preds.append(LinkPrediction(g.entity_id(x), 1.0, Explanation(
                    kind="path", confidence=1.0, steps=[hop_step(g, hop, f"{PROOF_TAG}: stated fact")], text=text)))
            return LPResult(query=(head, relation, tail), predictions=known_preds + preds)
        known = [g.label(x) for x in stored]
        if known and preds:
            # Link prediction looks for facts the graph is MISSING (the standard filtered setting, which the
            # evaluation relies on), so stored answers are hidden. Say so, or the top new answer reads as
            # if it replaced the stored one (USA capital -> New York while Washington, D.C. is stored).
            shown = ", ".join(known[:5]) + (f" and {len(known) - 5} more" if len(known) > 5 else "")
            preds[0].explanation.text = (
                f"ℹ️ Already stated in the graph, so not listed as predictions: **{shown}**. "
                f"The answers below are additional answers the graph does not contain yet.\n\n"
                + preds[0].explanation.text)
        return LPResult(query=(head, relation, tail), predictions=preds)

    def _lp_rules(self, src: int, rel: int, exclude: set[int], k: int, q: str) -> list[LinkPrediction]:
        st, rc = self._need(), self.cfg.reasoning
        if not st.rb.cp_by_head and not st.rb.uc_index:
            raise ModuleError("INTERNAL", f"No mined rules for {st.kg.name}. Run modules/expath/scripts/mine_rules.py "
                                          f"--dataset {st.kg.name} (or export a KGE, see README).")
        firings = apply_rules(st.g, st.rb, src, rel, rc)
        out = []
        for cand, conf in rank_candidates(firings, rc.aggregation, exclude, 3 * k):
            grounded = ground_firings(st.g, src, cand, firings[cand], rc.rules_shown)
            if not grounded and self.cfg.require_proof:
                continue
            ex = firings_explanation(st.g, f"**{st.g.label(cand)}** answers {q}: the mined rules below fire for it.",
                                     grounded, conf, rules_shown=rc.rules_shown, n_total=len(firings[cand]))
            out.append(LinkPrediction(st.g.entity_id(cand), conf, ex))
            if len(out) >= k:
                break
        return out

    def _lp_kge(self, src: int, rel: int, exclude: set[int], k: int, q: str) -> list[LinkPrediction]:
        st, rc = self._need(), self.cfg.reasoning
        scores = st.kge.score_all(src, rel)
        finite = np.isfinite(scores)
        if not finite.any():
            raise ModuleError("UNKNOWN_ENTITY", "The KGE has no embedding for this query.")
        z = scores[finite].max()
        probs = np.where(finite, np.exp(scores - z), 0.0)
        probs /= probs.sum()
        order = [int(i) for i in np.argsort(-scores) if finite[i] and int(i) not in exclude][: self.cfg.kge_candidates]
        out, unproven = [], 0
        budget = [self.cfg.kge_on_demand_budget]                 # shared across this query's candidates
        for rank, cand in enumerate(order, 1):
            ev = explain_triple(st.g, st.rb, src, rel, cand, rc, self.cfg.quality, self.cfg.mining.row_chunk,
                                budget=budget)
            if not ev.firings and self.cfg.require_proof:
                unproven += 1
                continue
            intro = (f"**{st.g.label(cand)}** is the embedding model's #{rank} answer to {q} "
                     f"(model probability {probs[cand]:.3f}). eXpath proves it with a rule from the graph:")
            ex = triple_explanation(st.g, intro, ev, rc.rules_shown, rc.critical_facts_shown)
            if unproven:
                were, them = ("s were", "them") if unproven > 1 else (" was", "it")
                ex.text += (f"\n\n<sub>{unproven} higher-ranked model answer{were} dropped: "
                            f"eXpath found no rule proving {them}.</sub>")
            out.append(LinkPrediction(st.g.entity_id(cand), float(probs[cand]), ex))
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------------ claim verification

    def verify_claim(self, claim: str) -> CVResult:
        st = self._need()
        if st.claim_mode == "multi_entity":
            return self._verify_multi(claim)
        return self._verify_triple(claim)

    def _verify_triple(self, claim: str) -> CVResult:
        """The parser returns the positive triple only, so a negated claim ("X is not the capital of Y")
        is checked as the positive fact and the verdict flipped, as in multi-entity mode."""
        # The parser's wording cues ("is the capital of") miss "is not the capital of", so it parses the
        # positive sentence; whether the claim was negated is decided on the original text below.
        positive = re.sub(r"\s+", " ", re.sub(r"\b(not|never)\b|n't\b", " ", claim, flags=re.IGNORECASE)).strip()
        res = self._verify_positive_triple(positive)
        res.claim = claim
        rest = claim
        for e in (res.triple[0], res.triple[2]) if res.triple else ():
            # "No Country for Old Men" is a title, not a negation
            rest = re.sub(re.escape(self._need().g.labels.get(e, e)), " ", rest, flags=re.IGNORECASE)
        if not is_negated(rest) or res.verdict == "NOT_ENOUGH_INFO":
            return res
        flipped = "REFUTED" if res.verdict == "SUPPORTED" else "SUPPORTED"
        ex = res.explanation
        ex.text = (f"**The claim is negated** (\"not\"), so eXpath checks the positive statement and reverses "
                   f"the verdict: the evidence below {'confirms' if flipped == 'REFUTED' else 'contradicts'} the "
                   f"positive statement, so the negated claim is **{flipped}**.{NL}{NL}" + ex.text)
        return CVResult(res.claim, res.triple, flipped, res.confidence, ex)

    def _verify_positive_triple(self, claim: str) -> CVResult:
        st, g, vc, rc = self._need(), self._need().g, self.cfg.verdict, self.cfg.reasoning
        parser_url = self.cfg.parser_url or get_service("claim_parser").base_url
        try:
            h_id, r_name, t_id = parse_claim(parser_url, claim, st.kg.name)["triple"]
        except ModuleError as e:
            if e.code == "INTERNAL" and "Cannot reach" in e.message:
                raise ModuleError("INTERNAL", f"The claim parser is not running ({parser_url}). Claims on "
                                              f"'{st.kg.name}' need it: start it with "
                                              f"`python scripts/start_services.py claim_parser`. (FactKG claims "
                                              f"don't use it: pick the factkg dataset for those.)") from e
            raise
        h, r, t = self._ent(h_id), self._rel(r_name), self._ent(t_id)
        triple = (h_id, r_name, t_id)

        if g.has_edge(h, r, t):
            step = hop_step(g, (h, r, t), f"{PROOF_TAG} (stated fact)")
            return CVResult(claim, triple, "SUPPORTED", 1.0, Explanation(
                kind="path", steps=[step], confidence=1.0,
                text=f"{g.hop_text((h, r, t))} is stated directly in the knowledge graph."))

        ev = explain_triple(g, st.rb, h, r, t, rc, self.cfg.quality, self.cfg.mining.row_chunk)
        claimed = f"**{g.hop_text((h, r, t))}**"
        if ev.confidence >= vc.support:
            ex = triple_explanation(g, f"{claimed} is not stored, but eXpath infers it (confidence "
                                       f"{ev.confidence:.2f} >= {vc.support}).", ev, rc.rules_shown, rc.critical_facts_shown)
            return CVResult(claim, triple, "SUPPORTED", ev.confidence, ex)

        others = [x for x in g.neighbors_by_rel(h, r).tolist() if x != t]
        if others and st.functional[r] >= vc.functional_ratio:
            steps = [hop_step(g, (h, r, x), f"{PROOF_TAG} (conflicting fact)") for x in others[:3]]
            text = (f"{claimed} conflicts with the graph: [{g.rel_short(r)}] has a single value for {st.functional[r]:.0%} of "
                    f"its subjects, and the graph states **{g.hop_text((h, r, others[0]))}**. "
                    f"eXpath's confidence for the claimed link is only {ev.confidence:.2f}.")
            return CVResult(claim, triple, "REFUTED", st.functional[r], Explanation(
                kind="path", steps=steps, text=text, confidence=st.functional[r]))

        if st.rb.cp_by_head or st.rb.uc_index:
            firings = apply_rules(g, st.rb, h, r, rc)
            alts = rank_candidates(firings, rc.aggregation, {h, t}, 1)
            if alts and alts[0][1] >= vc.refute_alt and alts[0][1] > ev.confidence:
                alt, conf = alts[0]
                grounded = ground_firings(g, h, alt, firings[alt], rc.rules_shown)
                if grounded:
                    ex = firings_explanation(
                        g, f"{claimed} has eXpath confidence {ev.confidence:.2f}; the graph instead entails "
                           f"**{g.label(alt)}** for {g.label(h)} —[{g.rel_short(r)}]→ ? with confidence {conf:.2f}.",
                        grounded, conf, rules_shown=rc.rules_shown)
                    return CVResult(claim, triple, "REFUTED", conf, ex)

        ex = triple_explanation(g, f"{claimed}: eXpath confidence {ev.confidence:.2f} is below {vc.support} "
                                   f"and no conflicting fact was found.", ev, rc.rules_shown, rc.critical_facts_shown)
        if not ex.steps:
            ex.steps.append(ExplanationStep(triple=None, note=f"No proof for {claimed}; searched {ev.n_paths} paths."))
        return CVResult(claim, triple, "NOT_ENOUGH_INFO", None, ex)

    def _verify_multi(self, claim: str) -> CVResult:
        st, g, vc, rc = self._need(), self._need().g, self.cfg.verdict, self.cfg.reasoning
        if st.linker is None and claim.strip() not in st.entity_sets:
            degree = {e: g.degree(i) for e, i in g.ent2idx.items()}
            st.linker = EntityLinker(g.labels, degree)
        q = analyse(claim, st.entity_sets or {}, st.linker, g.labels)
        if not q.entities:
            raise ModuleError("UNPARSEABLE_CLAIM", f"No knowledge-graph entity found in claim: {claim!r}")

        cache: dict[int, tuple[float, str]] = {}

        def rel_score(r: int) -> tuple[float, str]:
            if r not in cache:
                cache[r] = q.relevance(g.rel_base(r), st.lexicon)
            return cache[r]

        note = ""
        log = self.cfg.artifacts(st.kg.name) / "additions.jsonl"
        missing = [e for e in q.entities if e not in g.ent2idx]
        if missing and self.cfg.enrich.enabled:
            note = self._enrichment_note(enrich(g, missing, self.cfg.enrich, log, self._enriched))
            missing = [e for e in q.entities if e not in g.ent2idx]

        def run():
            ents = [g.ent2idx[e] for e in q.entities if e in g.ent2idx]
            return gather_evidence(g, st.rb, ents, rel_score, vc, rc, self.cfg.quality,
                                   single=len(q.entities) == 1, functional=st.functional)

        ev = run()
        needs_more = (ev.unlinked and not ev.conflicts) or (len(q.entities) == 1 and ev.existence is None)
        if self.cfg.enrich.enabled and needs_more:
            targets = ev.unlinked or q.entities
            summary = enrich(g, [e for e in targets if e not in self._enriched], self.cfg.enrich, log, self._enriched)
            if summary["entities_queried"]:
                note = (note + " " + self._enrichment_note(summary)).strip()
                ev = run()
        return decide(g, claim, q, ev, missing, note, vc.closed_world_existence)

    @staticmethod
    def _enrichment_note(summary: dict) -> str:
        if not summary["entities_queried"]:
            return ""
        txt = (f"Node addition: queried DBpedia for {', '.join(summary['entities_queried'])}; added "
               f"{summary['facts_added']} facts and {len(summary['new_nodes'])} new nodes (cited with their DBpedia source).")
        if summary["errors"]:
            txt += f" Retrieval errors: {'; '.join(summary['errors'][:2])}."
        return txt
