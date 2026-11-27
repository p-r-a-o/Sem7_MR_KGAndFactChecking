"""Pure, stdlib-only logic for the PyClause module: rule-file stats, ranking, verdicts, explanations.

Nothing here imports PyClause, so it is unit-testable in any environment. The PyClause calls live in
engine.py, which hands back the plain dataclasses defined below.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

from common.schema import Explanation, ExplanationStep, Triple

Direction = Literal["head", "tail"]
Grounding = tuple[Triple, ...]          # the KG facts that instantiate one rule body


# ----------------------------------------------------------------------------- engine results

@dataclass(frozen=True)
class Answer:
    """One candidate from PyClause's QAHandler, with the rules that predicted it (best first)."""
    candidate: str
    score: float
    rules: tuple[str, ...] = ()


@dataclass(frozen=True)
class TripleExplanation:
    """PyClause PredictionHandler output for one triple.

    rules[j] predicts the triple; groundings[j] lists that rule's groundings (each a tuple of facts).
    A triple no rule predicts has score 0.0 and no rules.
    """
    triple: Triple
    score: float
    rules: tuple[str, ...] = ()
    groundings: tuple[tuple[Grounding, ...], ...] = ()


# ----------------------------------------------------------------------------- rules

_ATOM = re.compile(r"([^\s(),]+)\(([^()]*)\)")
_VARIABLE = re.compile(r"^[A-Z]$")


@dataclass(frozen=True)
class RuleStats:
    predictions: int     # body groundings in the train graph
    support: int         # ... of which the head also holds
    confidence: float    # as written in the rule file (support / predictions, unsmoothed)

    @property
    def text(self) -> str:
        return f"conf {self.confidence:.3f}, {self.support} of {self.predictions} train predictions correct"


def rule_text(rule: str) -> str:
    """PyClause may hand back 'preds<TAB>support<TAB>conf<TAB>rule'; keep the rule part only."""
    return rule.split("\t")[-1].strip()


def rule_key(rule: str) -> str:
    """Whitespace-insensitive key, so rule strings from PyClause match rule-file lines."""
    return "".join(rule_text(rule).split())


def parse_rule_line(line: str) -> Optional[tuple[str, RuleStats]]:
    """AnyBURL/PyClause rule-file line 'preds<TAB>support<TAB>conf<TAB>rule' -> (rule, stats)."""
    parts = line.rstrip("\r\n").split("\t")
    if len(parts) != 4 or "<=" not in parts[3]:
        return None
    try:
        stats = RuleStats(int(parts[0]), int(parts[1]), float(parts[2]))
    except ValueError:
        return None
    return parts[3].strip(), stats


def load_rule_stats(path: Path) -> dict[str, RuleStats]:
    """rule_key -> stats for every parseable line. Used for display only; PyClause does the scoring."""
    stats: dict[str, RuleStats] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parsed = parse_rule_line(line)
            if parsed:
                stats[rule_key(parsed[0])] = parsed[1]
    return stats


def atoms(rule: str) -> tuple[tuple[str, tuple[str, ...]], tuple[tuple[str, tuple[str, ...]], ...]]:
    """'h(X,Y) <= b1(X,A), b2(A,Y)' -> (('h', ('X','Y')), (('b1', ('X','A')), ('b2', ('A','Y'))))."""
    head_text, _, body_text = rule_text(rule).partition("<=")
    head = _ATOM.findall(head_text)
    if not head:
        raise ValueError(f"Not a rule: {rule!r}")
    as_atom = lambda m: (m[0], tuple(a.strip() for a in m[1].split(",")))  # noqa: E731
    return as_atom(head[0]), tuple(as_atom(m) for m in _ATOM.findall(body_text))


def is_variable(arg: str) -> bool:
    return bool(_VARIABLE.match(arg))


def rule_kind(rule: str) -> str:
    """PyClause's rule families: B (cyclic), U_c / U_d (acyclic with a constant), Z (empty body)."""
    (_, (s, o)), body = atoms(rule)
    if not body:
        return "Z"
    if is_variable(s) and is_variable(o):
        return "B" if s != o else "U_xx"
    body_args = [a for _, args in body for a in args]
    return "U_c" if body_args and not is_variable(body_args[-1]) else "U_d"


def count_rule_kinds(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parsed = parse_rule_line(line)
            if parsed:
                try:
                    kind = rule_kind(parsed[0])
                except ValueError:
                    kind = "unparsed"
                counts[kind] = counts.get(kind, 0) + 1
    return counts


# ----------------------------------------------------------------------------- ranking & verdicts

def rank_of(answers: Sequence[Answer], candidate: str) -> Optional[int]:
    """1-based rank of `candidate` among answers, None if absent."""
    for i, a in enumerate(answers, 1):
        if a.candidate == candidate:
            return i
    return None


def decide_verdict(rank: Optional[int], n_candidates: int, support_rank: int, refute_rank: int,
                   min_candidates_for_refute: int) -> str:
    """Rank of the claimed entity among rule-derived candidates -> verdict.

    `rank is None` means no rule ranked it within top-k: that refutes only when the rules derived
    other answers, otherwise the graph simply has nothing to say about the query.
    """
    if rank is None:
        return "REFUTED" if n_candidates >= max(1, min_candidates_for_refute) else "NOT_ENOUGH_INFO"
    if rank <= support_rank:
        return "SUPPORTED"
    if rank > refute_rank:
        return "REFUTED"
    return "NOT_ENOUGH_INFO"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- explanation text

def label_of(entity: str, labels: dict[str, str]) -> str:
    return labels.get(entity, entity)


def disambiguate_labels(labels: dict[str, str]) -> dict[str, str]:
    """Append the entity ID to any label shared by several entities (display only)."""
    counts: dict[str, int] = {}
    for name in labels.values():
        counts[name] = counts.get(name, 0) + 1
    return {e: (f"{n} [{e}]" if counts[n] > 1 else n) for e, n in labels.items()}


def format_fact(fact: Triple, labels: dict[str, str]) -> str:
    h, r, t = fact
    return f"{label_of(h, labels)} --[{r}]--> {label_of(t, labels)}"


def format_grounding(facts: Sequence[Triple], labels: dict[str, str], start: Optional[str] = None) -> str:
    """Render a grounding as a path when the facts chain from `start` (in any order), else as a list."""
    if not facts:
        return "no body atoms"
    if start is not None:
        current, parts, left = start, [label_of(start, labels)], list(facts)
        while left:
            for i, (h, r, t) in enumerate(left):
                if h == current:
                    parts.append(f" --[{r}]--> {label_of(t, labels)}")
                    current = t
                    break
                if t == current:
                    parts.append(f" --[{r}, inverse]--> {label_of(h, labels)}")
                    current = h
                    break
            else:
                break
            left.pop(i)
        if not left:
            return "".join(parts)
    return "; ".join(format_fact(f, labels) for f in facts)


def _rule_label(rule: str, stats: dict[str, RuleStats]) -> str:
    s = stats.get(rule_key(rule))
    return f"{rule_text(rule)} ({s.text})" if s else rule_text(rule)


def grounding_steps(rule: str, grounding: Grounding, labels: dict[str, str],
                    max_steps: int = 8) -> list[ExplanationStep]:
    """One step per body fact; annotated with the body atom when the counts line up."""
    try:
        body = atoms(rule)[1]
    except ValueError:
        body = ()
    names = [f"{rel}({','.join(args)})" for rel, args in body] if len(body) == len(grounding) else []
    steps = []
    for i, fact in enumerate(grounding[:max_steps]):
        note = format_fact(fact, labels) + (f" grounds {names[i]}" if names else "")
        steps.append(ExplanationStep(triple=fact, note=note))
    if len(grounding) > max_steps:
        steps.append(ExplanationStep(triple=None, note=f"... {len(grounding) - max_steps} more facts"))
    return steps


def rule_explanation(expl: TripleExplanation, stats: dict[str, RuleStats], labels: dict[str, str],
                     intro: str, aggregation: str, start: Optional[str] = None,
                     max_steps: int = 8, rules_mentioned: int = 3) -> Optional[Explanation]:
    """Best rule + its first grounding, plus the runner-up rules. None if no rule predicts the triple."""
    if not expl.rules:
        return None
    best_rule = expl.rules[0]
    groundings = expl.groundings[0] if expl.groundings else ()
    grounding = groundings[0] if groundings else ()
    best_stats = stats.get(rule_key(best_rule))
    text = (f"{intro} Derived by the rule below ({best_stats.text if best_stats else 'PyClause rule'}), "
            f"grounded as: {format_grounding(grounding, labels, start)}.")
    if len(groundings) > 1:
        text += f" This rule has {len(groundings)} groundings for the triple; the first is shown."
    try:
        (_, (s, o)), _body = atoms(best_rule)
        if not (is_variable(s) and is_variable(o)):
            other = o if is_variable(s) else s
            text += (f" The rule names {label_of(other, labels)} in its own head, so the body only has to "
                     f"hold for the rule to conclude it.")
    except ValueError:
        pass
    others = expl.rules[1:rules_mentioned]
    if others:
        text += " Also derived by: " + " | ".join(_rule_label(r, stats) for r in others) + "."
    if len(expl.rules) > 1:
        text += (f" {len(expl.rules)} rules predict this triple; aggregated score {expl.score:.3f} "
                 f"({aggregation}).")
    steps = grounding_steps(best_rule, grounding, labels, max_steps) or [
        ExplanationStep(triple=None, note="Rule has no body atoms.")]
    return Explanation(kind="rule", steps=steps, text=text, rule=rule_text(best_rule),
                       confidence=clamp01(expl.score))


def answer_rules_explanation(answer: Answer, stats: dict[str, RuleStats], intro: str,
                             rules_mentioned: int = 3) -> Explanation:
    """Fallback when PredictionHandler returns no grounding (e.g. a rule type it does not apply)."""
    rules = answer.rules[:rules_mentioned]
    text = intro + (" Predicted by: " + " | ".join(_rule_label(r, stats) for r in rules) + "."
                    if rules else " PyClause returned no predicting rule for this candidate.")
    return Explanation(kind="rule", steps=[ExplanationStep(triple=None, note=_rule_label(r, stats)) for r in rules],
                       text=text, rule=rule_text(rules[0]) if rules else None,
                       confidence=clamp01(answer.score))


def direct_fact_explanation(triple: Triple, labels: dict[str, str]) -> Explanation:
    h, r, t = triple
    return Explanation(
        kind="rule", confidence=1.0, rule=None,
        steps=[ExplanationStep(triple=triple, note="Fact present in the training graph.")],
        text=f"({label_of(h, labels)}, {r}, {label_of(t, labels)}) is stated directly in the knowledge graph.")


def no_rule_explanation(reason: str) -> Explanation:
    return Explanation(kind="rule", steps=[], text=reason, confidence=None)


def top_candidates_text(answers: Sequence[Answer], labels: dict[str, str], n: int = 3) -> str:
    return ", ".join(f"{label_of(a.candidate, labels)} ({a.score:.3f})" for a in answers[:n])
