"""PGR (Programmatic Graph Reasoning, Hao & Wu, Findings of EMNLP 2025) behind the shared interface.

Claim verification only: an LLM writes a small program of MATCH / SEARCH / VERIFY calls for the claim,
the program runs against the knowledge graph, and VERIFY's result is the verdict. The proof is the
program, the result of every call, and the graph facts each call matched (all checkable in the graph).
PGR needs the claim's entity set (FactKG ships one per claim), so it is set up for FactKG.
"""
import json
from typing import Optional

from common.datasets.base import KnowledgeGraph
from common.errors import ModuleError
from common.paths import data_root
from common.schema import CVResult, Explanation, ExplanationStep, LPResult
from modules.pgr.config import PGRConfig
from modules.pgr.llm import ChatClient, DailyQuotaExhausted, LLMError
from modules.pgr.reasoning import PGRGraph, Reasoner, Trace, extract_program, prompt, run_program


RETRY_SUFFIX = ("\n\nReply with ONLY the program for the task above, inside one ```python code block, "
                "with no explanation or reasoning text.")


def load_entity_sets(dataset: str) -> dict[str, list[str]]:
    """claim text -> FactKG's entity set, from data/claims/<dataset>/*.jsonl."""
    out: dict[str, list[str]] = {}
    for p in sorted((data_root() / "claims" / dataset).glob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("entity_set"):
                    out[d["claim"].strip()] = d["entity_set"]
    return out


class PGRModule:
    name = "PGR"

    def __init__(self, cfg: Optional[PGRConfig] = None):
        self.cfg = cfg or PGRConfig()
        self.llm = ChatClient(self.cfg.llm, self.cfg.cache_path())
        self.kg: Optional[KnowledgeGraph] = None
        self.reasoner: Optional[Reasoner] = None
        self.entity_sets: dict[str, list[str]] = {}

    def load(self, kg: KnowledgeGraph) -> None:
        self.kg = kg
        self.reasoner = Reasoner(PGRGraph(kg), self.llm, self.cfg.llm.max_prompt_chars, self.cfg.max_search_results)
        self.entity_sets = load_entity_sets(kg.name)
        print(f"[PGR] {kg.name}: {len(kg.neighbors)} entities, {len(self.entity_sets)} claims with entity sets, "
              f"LLM {self.cfg.llm.model}, {len(self.llm.cache)} cached replies", flush=True)

    def predict_link(self, head, relation, tail, k: int = 10) -> LPResult:
        raise ModuleError("BAD_REQUEST", "PGR verifies claims only; it does no link prediction. "
                                         "Use eXpath (or another module) for the Link prediction tab.")

    # ------------------------------------------------------------------ claim verification

    def verify_claim(self, claim: str) -> CVResult:
        if self.kg is None or self.reasoner is None:
            raise ModuleError("NOT_LOADED", "PGR: call load() first.")
        entities = self.entity_sets.get(claim.strip())
        if entities is None:
            raise ModuleError("UNPARSEABLE_CLAIM", "PGR needs the claim's entity set, which FactKG provides for "
                                                   "its own claims. Pick a sample claim, or paste a FactKG "
                                                   "claim exactly as written (dataset: factkg).")
        trace = self.trace(claim, entities)
        return self.result(claim, entities, trace)

    def trace(self, claim: str, entities: list[str]) -> Trace:
        """Generate the program (one LLM call, cached) and run it (MATCH/SEARCH may call the LLM too)."""
        text = prompt("prompt_w_entity.txt").replace("<<<<S>>>>", claim).replace("<<<<ENTITIES>>>>", str(entities))
        try:
            code = extract_program(self.llm.complete(text))
            if "def program():" not in code:
                # "Thinking" models sometimes reason aloud until the output limit and never write the
                # program. Ask once more for the code only (ours; the reference just records an error).
                code = extract_program(self.llm.complete(text + RETRY_SUFFIX))
            return run_program(code, self.reasoner, claim)
        except DailyQuotaExhausted as e:
            raise ModuleError("INTERNAL", str(e)) from e
        except LLMError as e:
            raise ModuleError("INTERNAL", f"LLM call failed: {e}") from e

    def result(self, claim: str, entities: list[str], trace: Trace) -> CVResult:
        labels = self.kg.entity_labels if self.kg else {}
        name = lambda e: labels.get(e, e) if isinstance(e, str) else str(e)   # noqa: E731
        lines = [f"PGR: the LLM (`{self.cfg.llm.model}`) wrote a program for the claim, which was run against "
                 f"the knowledge graph. Claim entities (given by FactKG): "
                 f"{', '.join(name(e) for e in entities)}.", "", "```python", trace.program or "(no program)", "```"]
        steps: list[ExplanationStep] = []
        for i, c in enumerate(trace.calls, 1):
            if c.func == "SEARCH":
                found = c.result or []
                shown = ", ".join(name(e) for e in found[:5]) + (f" … ({len(found)} in total)" if len(found) > 5 else "")
                outcome = f"found {shown}" if found else "found nothing"
            else:
                outcome = f"**{c.result}**"
            lines.append(f"{i}. `{c.line}` → {outcome}" + (f" ({c.how})" if c.how else ""))
            for t in c.evidence:
                if len(steps) < self.cfg.evidence_shown and t not in {s.triple for s in steps}:
                    steps.append(ExplanationStep(triple=t, note=f"{c.func} in line {i} matched this fact · "
                                                                f"source: knowledge graph"))
        if trace.error:
            verdict = "NOT_ENOUGH_INFO"
            lines.append(f"\nThe program could not be run to the end ({trace.error}), so there is no verdict.")
        else:
            verdict = "SUPPORTED" if trace.predicted else "REFUTED"
            lines.append(f"\n`predicted` = **{trace.predicted}** → **{verdict}**.")
        lines.append("\n<sub>The LLM decides which graph facts match the claim's wording; every fact listed below "
                     "is in the graph and can be checked there.</sub>")
        if not steps:
            steps.append(ExplanationStep(triple=None, note="No graph fact matched any step of the program."))
        ex = Explanation(kind="subgraph", steps=steps, text="\n".join(lines), rule=None, confidence=None)
        return CVResult(claim=claim, triple=None, verdict=verdict, confidence=None, explanation=ex)
