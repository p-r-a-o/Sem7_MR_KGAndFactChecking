"""PGR's graph reasoning functions and a safe interpreter for LLM-written programs.

Port of hydragon/PGR `kg_program_bi_reverse.py` (Apache-2.0), same semantics, with three changes:
  1. The program is NOT run with eval(). A small interpreter accepts only `x = MATCH(...)`,
     `x = SEARCH(...)` and `x = VERIFY(...)` with literal / variable / and-or-not arguments.
  2. MATCH keeps only LLM-chosen triples that are really among the graph's candidate triples, so the
     proof cites facts that exist (the reference accepted any non-empty LLM answer).
  3. Every call is recorded (`Trace`) with the graph facts it relied on, which becomes the proof.
Graph view: PGR expects graph[entity][relation] -> [entities] with reverse edges as '~relation'.
`PGRGraph` builds that view per entity from the shared KnowledgeGraph.neighbors (inverse = '^-1').
"""
import ast
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from common.datasets.base import INVERSE_SUFFIX, KnowledgeGraph
from modules.pgr.llm import ChatClient

PROMPTS = Path(__file__).resolve().parent / "prompts"
Triple = tuple[str, str, str]


def prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


class PGRGraph:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self._rels = lru_cache(maxsize=4096)(self._build)

    def __contains__(self, e: str) -> bool:
        return e in self.kg.neighbors

    def _build(self, e: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for r, n in self.kg.neighbors.get(e, ()):
            key = "~" + r[: -len(INVERSE_SUFFIX)] if r.endswith(INVERSE_SUFFIX) else r
            out.setdefault(key, []).append(n)
        return out

    def rels(self, e: str) -> dict[str, list[str]]:
        return self._rels(e) if e in self else {}


# ----------------------------------------------------------------------------- trace (the proof)

@dataclass
class Call:
    line: str                       # the program line as executed
    func: str
    result: Any
    evidence: list[Triple] = field(default_factory=list)   # graph facts (stored direction) it relied on
    how: str = ""                   # "exact relation", "LLM matched", "no candidates", ...


@dataclass
class Trace:
    program: str = ""
    calls: list[Call] = field(default_factory=list)
    predicted: Optional[bool] = None
    error: str = ""


# ----------------------------------------------------------------------------- LLM-backed helpers

def parse_json_list(text: str) -> list:
    """Reference: extract_json_from_string (```json fence, else whole text). Safe literal parsing only."""
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    raw = (m.group(1) if m else text).strip()
    for parse in (json.loads, ast.literal_eval):
        try:
            val = parse(raw)
            return val if isinstance(val, list) else []
        except (ValueError, SyntaxError, TypeError):
            continue
    return []


class Reasoner:
    def __init__(self, graph: PGRGraph, llm: ChatClient, max_prompt_chars: int, max_search_results: int):
        self.g, self.llm = graph, llm
        self.max_prompt_chars = max_prompt_chars
        self.max_search_results = max_search_results

    def _ask_triples(self, template: str, triplets: list, claim: str, target: list) -> list:
        """MATCH/SEARCH prompt -> LLM-selected triplets. Reference skips prompts over 8192 chars."""
        if not triplets:
            return []
        text = (prompt(template).replace("<<<<LIST>>>>", json.dumps(triplets, ensure_ascii=False))
                .replace("<<<<CLAIM>>>>", claim).replace("<<<<TARGET>>>>", json.dumps(target, ensure_ascii=False)))
        if len(text) > self.max_prompt_chars:
            return []
        return [t for t in parse_json_list(self.llm.complete(text)) if isinstance(t, (list, tuple)) and len(t) == 3]

    # ---- SEARCH: (h, r, None) or (None, r, t) -> candidate entities
    def search(self, triplet, claim: str, call: Call) -> list[str]:
        head, relation, tail = triplet
        known, missing_head = (tail, True) if head is None else (head, False)
        if not isinstance(known, str) or known not in self.g or not isinstance(relation, str):
            call.how = "entity not in graph"
            return []
        rels = self.g.rels(known)
        direct = ("~" + relation) if missing_head else relation
        if direct in rels:                                     # exact relation name: no LLM call
            ents = rels[direct][: self.max_search_results]
            call.how = "exact relation"
            call.evidence = [(e, relation, known) if missing_head else (known, relation, e) for e in ents]
            return ents
        # one example triple per relation, both directions (as the reference builds triplet_set_0)
        cands = ([[known, r, ns[0]] for r, ns in rels.items() if not r.startswith("~")] +
                 [[ns[0], r[1:], known] for r, ns in rels.items() if r.startswith("~")])
        target = ["Unknown", relation, known] if missing_head else [known, relation, "Unknown"]
        picked = self._ask_triples("search_direct_prompt.txt", cands, claim, target)
        ents: list[str] = []
        for t in picked:
            r = str(t[1])
            # the chosen candidate shows the direction: [known, r, x] is a forward edge, [x, r, known] a reverse one
            order = (r, "~" + r) if str(t[0]) == known else ("~" + r, r) if str(t[2]) == known else \
                ((r, "~" + r) if not missing_head else ("~" + r, r))
            for key in order:
                if key in rels:
                    for e in rels[key][: self.max_search_results]:
                        ents.append(e)
                        fwd = (known, r, e) if not key.startswith("~") else (e, r, known)
                        call.evidence.append(fwd)
                    break
        call.how = "LLM chose relation(s): " + ", ".join(sorted({str(t[1]) for t in picked})) if picked else \
            "LLM found no matching relation"
        return list(dict.fromkeys(ents))[: self.max_search_results]

    # ---- MATCH: does (h, r, t) hold? h / t may be lists from SEARCH
    def match(self, triplet, claim: str, call: Call) -> bool:
        head, relation, tail = triplet
        heads = head if isinstance(head, list) else [head]
        tails = tail if isinstance(tail, list) else [tail]
        possible: list[Triple] = []
        for e1 in heads:
            if not isinstance(e1, str) or e1 not in self.g:
                continue
            for r, ns in self.g.rels(e1).items():
                for e2 in tails:
                    if e2 in ns:
                        possible.append((e1, r, e2) if not r.startswith("~") else (e2, r[1:], e1))
        possible = list(dict.fromkeys(possible))
        exact = [t for t in possible if t[1] == relation]
        if exact:
            call.how, call.evidence = "exact relation", exact
            return True
        if not possible:
            call.how = "no fact links these entities"
            return False
        picked = self._ask_triples("match_direct_prompt.txt", [list(t) for t in possible], claim,
                                   [heads, relation, tails])
        real = [t for t in (tuple(map(str, p)) for p in picked) if t in set(possible)]   # change 2
        call.evidence = real
        call.how = "LLM matched " + ", ".join(t[1] for t in real) if real else \
            f"LLM found none of the {len(possible)} linking facts means '{relation}'"
        return bool(real)


# ----------------------------------------------------------------------------- safe interpreter

_LINE = re.compile(r"^(\w+)\s*=\s*(MATCH|SEARCH|VERIFY)\((.*)\)\s*$")


def _value(node: ast.AST, env: dict) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"undefined variable {node.id!r}")
        return env[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_value(e, env) for e in node.elts)
    if isinstance(node, ast.BoolOp):
        vals = [bool(_value(v, env)) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not bool(_value(node.operand, env))
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, right = node.left, node.comparators[0]
        op = node.ops[0]
        # `entity_1 is not None` asks "did SEARCH find anything?". SEARCH returns a list ([] if nothing),
        # so literal Python would say True for an empty result; we read it as the intended emptiness test.
        if isinstance(right, ast.Constant) and right.value is None and isinstance(op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
            found = bool(_value(left, env))
            return found if isinstance(op, (ast.IsNot, ast.NotEq)) else not found
        if isinstance(op, (ast.Eq, ast.NotEq)):
            same = _value(left, env) == _value(right, env)
            return same if isinstance(op, ast.Eq) else not same
    raise ValueError(f"unsupported expression: {ast.dump(node)[:80]}")


def extract_program(reply: str) -> str:
    """Reference: the ```python fence if present, else the raw reply. Also: a fence that holds the
    program wins over other fences, and a bare `def program():` anywhere in the reply is taken from
    there on (models that "think aloud" put text before it)."""
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", reply, re.DOTALL)
    for f in fences:
        if "def program():" in f:
            return f.strip()
    i = reply.find("def program():")
    if i >= 0:
        return reply[i:].strip()
    return (fences[0] if fences else reply).strip()


def run_program(code: str, reasoner: Reasoner, claim: str) -> Trace:
    trace = Trace(program=code)
    body = re.search(r"def program\(\):\s*\n(.*)", code, re.DOTALL)
    if not body:
        trace.error = "the LLM did not return a program"
        return trace
    env: dict[str, Any] = {}
    try:
        for raw in body.group(1).splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "predicted" in env:
                break                  # anything after the verdict line (a stray "}" or "end") is ignored
            m = _LINE.match(line)
            if not m:
                if line.startswith("return"):
                    continue
                raise ValueError(f"line not allowed: {line[:80]}")
            var, func, args = m.groups()
            arg = _value(ast.parse(args, mode="eval").body, env)
            call = Call(line=line, func=func, result=None)
            if func == "VERIFY":
                call.result = bool(arg)
            elif not (isinstance(arg, tuple) and len(arg) == 3):
                raise ValueError(f"{func} needs a (head, relation, tail) triple: {line[:80]}")
            elif func == "SEARCH":
                call.result = reasoner.search(arg, claim, call)
            else:
                call.result = reasoner.match(arg, claim, call)
            env[var] = call.result
            trace.calls.append(call)
        if "predicted" not in env:
            raise ValueError("the program never sets 'predicted'")
        trace.predicted = bool(env["predicted"])
    except (ValueError, SyntaxError) as e:
        trace.error = str(e)
    return trace
