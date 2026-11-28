"""PGR module: interpreter safety, MATCH/SEARCH semantics, proof facts, cache. No API key needed
(a fake LLM answers from a script)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.datasets.base import KnowledgeGraph  # noqa: E402
from common.datasets.files import build_neighbors  # noqa: E402
from modules.pgr.config import LLMConfig, PGRConfig  # noqa: E402
from modules.pgr.llm import ChatClient  # noqa: E402
from modules.pgr.module import PGRModule  # noqa: E402
from modules.pgr.reasoning import PGRGraph, Reasoner, parse_json_list, run_program  # noqa: E402

TRAIN = [("Ab_Klink", "birthPlace", "Stellendam"), ("Ab_Klink", "almaMater", "Erasmus_University_Rotterdam"),
         ("Alfredo_Zitarrosa", "deathPlace", "Uruguay"), ("Alfredo_Zitarrosa", "birthPlace", "Uruguay"),
         ("Uruguay", "leader", "Tabaré_Vázquez"), ("Stellendam", "country", "Netherlands")]


def kg() -> KnowledgeGraph:
    ents = sorted({x for h, _, t in TRAIN for x in (h, t)})
    return KnowledgeGraph(name="toy", train=TRAIN, valid=[], test=[], relations=sorted({r for _, r, _ in TRAIN}),
                          entity_labels={e: e.replace("_", " ") for e in ents}, neighbors=build_neighbors(TRAIN))


class FakeLLM(ChatClient):
    """Answers prompts by substring rules; records the prompts it saw."""
    def __init__(self, rules: list[tuple[str, str]]):
        super().__init__(LLMConfig(), None)
        self.rules, self.seen = rules, []

    def complete(self, prompt: str) -> str:
        self.seen.append(prompt)
        tail = prompt.rsplit("Your Task", 1)[-1]          # match on the task part, not the few-shot examples
        for needle, reply in self.rules:
            if needle in tail:
                self.calls += 1
                return reply
        return "[]"


def reasoner(llm) -> Reasoner:
    return Reasoner(PGRGraph(kg()), llm, max_prompt_chars=8192, max_search_results=50)


def test_exact_relation_needs_no_llm_and_cites_the_fact():
    llm = FakeLLM([])
    tr = run_program("def program():\n    result_1 = MATCH(('Ab_Klink', 'birthPlace', 'Stellendam'))\n"
                     "    predicted = VERIFY(result_1)", reasoner(llm), "Ab Klink was born in Stellendam.")
    assert tr.predicted is True and llm.calls == 0
    assert tr.calls[0].evidence == [("Ab_Klink", "birthPlace", "Stellendam")]


def test_llm_match_keeps_only_real_triples():
    # the LLM names one real linking fact and one invented one: only the real one may count as proof
    llm = FakeLLM([("died in", json.dumps([["Alfredo_Zitarrosa", "deathPlace", "Uruguay"],
                                           ["Alfredo_Zitarrosa", "diedIn", "Uruguay"]]))])
    tr = run_program("def program():\n    result_1 = MATCH(('Alfredo_Zitarrosa', 'died in', 'Uruguay'))\n"
                     "    predicted = VERIFY(result_1)", reasoner(llm), "Alfredo Zitarrosa died in Uruguay.")
    assert tr.predicted is True
    assert tr.calls[0].evidence == [("Alfredo_Zitarrosa", "deathPlace", "Uruguay")]

    llm = FakeLLM([("died in", json.dumps([["Alfredo_Zitarrosa", "diedIn", "Uruguay"]]))])   # invented only
    tr = run_program("def program():\n    result_1 = MATCH(('Alfredo_Zitarrosa', 'died in', 'Uruguay'))\n"
                     "    predicted = VERIFY(result_1)", reasoner(llm), "x")
    assert tr.predicted is False and tr.calls[0].evidence == []


def test_search_then_match_multi_hop_and_negation():
    llm = FakeLLM([])
    code = ("def program():\n    entity_1 = SEARCH((None, 'leader', 'Tabaré_Vázquez'))\n"
            "    result_1 = MATCH(('Alfredo_Zitarrosa', 'deathPlace', entity_1))\n"
            "    predicted = VERIFY(not result_1)")
    tr = run_program(code, reasoner(llm), "Zitarrosa did not die in a country led by Tabaré Vázquez.")
    assert tr.calls[0].result == ["Uruguay"] and tr.calls[1].result is True
    assert tr.predicted is False                       # negated claim, positive fact holds -> REFUTED


def test_existence_check_and_trailing_junk():
    found = ("def program():\n    entity_1 = SEARCH(('Ab_Klink', 'birthPlace', None))\n"
             "    predicted = VERIFY(entity_1 is not None)\nend")
    none = ("def program():\n    entity_1 = SEARCH(('Ab_Klink', 'spouse', None))\n"
            "    predicted = VERIFY(entity_1 is not None)\n}")
    assert run_program(found, reasoner(FakeLLM([])), "x").predicted is True
    tr = run_program(none, reasoner(FakeLLM([])), "x")       # nothing found: [] must not count as "not None"
    assert tr.predicted is False and not tr.error


def test_interpreter_rejects_anything_but_the_three_functions():
    for bad in ("def program():\n    x = __import__('os')\n    predicted = VERIFY(True)",
                "def program():\n    result_1 = MATCH(open('f').read())\n    predicted = VERIFY(result_1)",
                "def program():\n    import os\n    predicted = VERIFY(True)"):
        tr = run_program(bad, reasoner(FakeLLM([])), "x")
        assert tr.predicted is None and tr.error


def test_parse_json_list():
    assert parse_json_list('```json\n[["a", "r", "b"]]\n```') == [["a", "r", "b"]]
    assert parse_json_list("no json here") == []


def test_extract_program_from_thinking_text():
    from modules.pgr.reasoning import extract_program
    reply = ("Let me think. Triple 1 is `X | architect | Y`.\n```text\nnot code\n```\nSo:\n"
             "```python\ndef program():\n    result_1 = MATCH(('A', 'r', 'B'))\n    predicted = VERIFY(result_1)\n```")
    assert extract_program(reply).startswith("def program():")
    bare = "Wait, let's check...\ndef program():\n    predicted = VERIFY(True)"
    assert extract_program(bare).startswith("def program():")


def test_retry_when_first_reply_has_no_program():
    m = PGRModule(PGRConfig())
    prog = "```python\ndef program():\n    result_1 = MATCH(('Ab_Klink', 'birthPlace', 'Stellendam'))\n" \
           "    predicted = VERIFY(result_1)\n```"
    m.llm = FakeLLM([("ONLY the program", prog),                      # the retry prompt
                     ("# Entities", "Wait, let's check the triples first... (output limit reached)")])
    m.load(kg())
    m.reasoner.llm = m.llm
    tr = m.trace("Ab Klink was born in Stellendam.", ["Ab_Klink", "Stellendam"])
    assert tr.predicted is True and not tr.error and m.llm.calls == 2


def test_daily_quota_detection():
    from modules.pgr.llm import is_daily_quota
    per_minute = ('{"error": {"code": 429, "message": "You exceeded your current quota. Free tier: requests '
                  'per day and per minute apply.", "details": [{"violations": [{"quotaId": '
                  '"GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "quotaValue": "5"}]}, '
                  '{"retryDelay": "37s"}]}}')
    per_day = '{"details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}'
    assert not is_daily_quota(per_minute)              # mentions "per day" in prose, but the quota is per minute
    assert is_daily_quota(per_day)
    assert not is_daily_quota("Resource exhausted. Please retry in 12.5s.")


def test_cache_survives_restart(tmp_path):
    path = tmp_path / "cache.jsonl"
    c = ChatClient(LLMConfig(), path)
    c._store(c._key("hello"), "world")
    c2 = ChatClient(LLMConfig(), path)
    assert c2.complete("hello") == "world" and c2.cache_hits == 1 and c2.calls == 0


def test_module_verdict_and_proof_steps():
    m = PGRModule(PGRConfig())
    m.llm = FakeLLM([("Target Triplet", json.dumps([["Ab_Klink", "birthPlace", "Stellendam"]])),   # MATCH prompt
                     ("# Entities", "```python\ndef program():\n    result_1 = MATCH(('Ab_Klink', 'born in', "
                                    "'Stellendam'))\n    predicted = VERIFY(result_1)\n```")])         # program prompt
    m.load(kg())
    m.reasoner.llm = m.llm
    res = m.result("Ab Klink was born in Stellendam.", ["Ab_Klink", "Stellendam"],
                   m.trace("Ab Klink was born in Stellendam.", ["Ab_Klink", "Stellendam"]))
    assert res.verdict == "SUPPORTED"
    assert [s.triple for s in res.explanation.steps] == [("Ab_Klink", "birthPlace", "Stellendam")]
    assert "MATCH" in res.explanation.text
