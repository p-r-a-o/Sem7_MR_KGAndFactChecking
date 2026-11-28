# PGR module (LLM-based claim verification)

Port of **PGR — Programmatic Graph Reasoning** (Hao & Wu, *Fact Verification on Knowledge Graph via
Programmatic Graph Reasoning*, Findings of EMNLP 2025; reference code
[hydragon/PGR](https://github.com/hydragon/PGR), Apache-2.0, see `LICENSE-PGR`). Prompts in `prompts/` are
copied unchanged from the reference.

For a claim, an LLM writes a small program over three functions, and the program is run on the graph:

| Function | Does | Uses the LLM? |
|---|---|---|
| `MATCH((h, r, t))` | is there a fact linking h and t that means r? | only if no fact has exactly relation r: the LLM picks which linking facts mean r |
| `SEARCH((h, r, None))` | the entities reached from h by relation r (or the reverse) | only if r is not an exact relation name: the LLM picks the relation(s) |
| `VERIFY(expr)` | the final True / False (with `and`, `or`, `not`) | no |

`predicted = True` → SUPPORTED, `False` → REFUTED. **The proof** is the program, the result of each line,
and every graph fact MATCH / SEARCH relied on (all of which exist in the graph and can be checked).

```python
def program():
    entity_1 = SEARCH((None, 'leader', 'Artur_Rasizade'))
    result_1 = MATCH(('"Hüseyin Bütüner and Hilmi Güner"', 'design', entity_1))
    predicted = VERIFY(result_1)
```

## Setup and use
Needs only the Python standard library and a **Gemini API key** (free tier; Google AI Studio → Get API key).
```powershell
$env:GEMINI_API_KEY = "<your key>"            # per terminal; never put the key in a file in the repo
python modules/pgr/scripts/list_models.py     # model ids your key can use (pick a free one in AI Studio)
python modules/pgr/api.py                     # server on port 8105 (services.yaml); --model, --rpm
python modules/pgr/scripts/run_eval.py        # evaluation within the quota, resumable (below)
```
Environment variables: `GEMINI_API_KEY` (required), `PGR_MODEL` (default `gemini-2.5-flash`), `PGR_RPM`
(requests per minute, default 4: keep it under the model's RPM in AI Studio), `PGR_LLM_URL` (any OpenAI-compatible chat endpoint; default Gemini's).
In the app: Module **PGR**, Dataset **factkg**, Claim verification tab, a FactKG claim (sample list or pasted
exactly: PGR needs FactKG's entity set for the claim). Link prediction is not supported.

## Free tier: cost and quota
The paper reports 2.2 LLM calls and ~2,700 tokens per claim on average. Every reply is cached in
`artifacts/pgr/llm_cache.jsonl` (keyed by model + prompt), so repeating a claim, a crash or a rerun costs
nothing. The client paces itself under `PGR_RPM`, waits out per-minute 429s, and stops cleanly when the
**daily** quota is used up.

`run_eval.py` draws `--per-type` claims (default 60) of each reasoning type from the FactKG test split
(seed 0) and visits them round-robin, so wherever the quota stops a run the finished set is balanced.
Results append to `artifacts/pgr/eval_test_<model>[_<reasoning>].jsonl` (one file per model, so different
models are never pooled into one score); rerun the next day to continue. `--pool` continues a sample across
models (a new model skips claims another model already finished; `--summary --pool` combines all files and
says which model answered how many, per type). A pooled score is not any single model's accuracy, so report
it as such; `compare_expath.py --pool` runs eXpath on the same pooled claims.

**Quotas are per model.** AI Studio's rate-limit page lists each model's requests per minute (RPM; free tier:
5 for the Gemini Flash models, 10-15 for Flash Lite, 30 for Gemma) and tokens per minute; keep `PGR_RPM`
below the RPM. A 429 is treated as the daily limit only when Google's error names a per-day quota
(`quotaId` containing `PerDay`); otherwise the client waits and retries. Pick one model for the whole
sample and keep it. Daily limits are not shown on that page; Google's 429 names them, e.g.
`gemini-3.5-flash`: `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, limit **20** requests/day
(~12 claims), far too few for a sample of hundreds. Gemma models (e.g. `gemma-4-31b-it`) are served without system
instructions (handled automatically) and may not accept `PGR_REASONING`; unset it for them. `--summary` prints
accuracy per type; `scripts/compare_expath.py` runs eXpath on exactly the finished claims.
Claim types: negation if negated, else existence, else multi-hop, else conjunction, else one-hop.

## Differences from the reference implementation
1. **No `eval()`** of LLM output: a small interpreter accepts only `x = MATCH|SEARCH|VERIFY(...)` lines with
   literal, variable and and/or/not arguments. Anything else is an error (no verdict).
2. **MATCH keeps only real facts**: LLM-chosen triples are kept only if they are among the graph's actual
   linking facts (the reference accepted any non-empty answer). Keeps every proof checkable.
3. **SEARCH direction** comes from the triple the LLM picked (the reference tried forward first).
4. **Graph**: our FactKG DBpedia subgraph (`data/factkg`, see `scripts/prepare_factkg.py`), which caps
   very-high-degree entities at 2,000 facts; the paper used FactKG's full DBpedia pickle.
5. **LLM**: a free Gemini model instead of GPT-4o; `max_tokens` 4096 instead of 1024 because Gemini
   "thinking" models spend output tokens on reasoning first. Temperature 0.2 / top-p 0.1 as in the paper.
6. Fixed a reference bug: `test_final.py` called `execute_program` with 4 arguments, which takes 5.
7. **One retry for a missing program**: "thinking" models sometimes reason aloud and stop before writing
   the program (seen with `gemini-3.5-flash`). If a reply has no `def program():`, the prompt is sent once
   more asking for the code only. A program found anywhere in a reply (not only in a code fence) is used.
   `PGR_REASONING=low` (sent as `reasoning_effort`) reduces the thinking; unset = model default.
   `run_eval.py --retry-errors` re-runs claims that ended without a verdict.

The paper's numbers (FactKG test, all 9,041 claims): GPT-4o 86.82%, DeepSeek-V3 84.18%, GPT-4o-mini 72.62%.
Ours are on a balanced sample with a free model, so report them as such, with the sample size.

## Measured (FactKG test split, balanced sample, seed 0)
257 claims (about 51 per reasoning type): 250 answered by `gemini-3.5-flash-lite` (`PGR_REASONING=low`),
7 by `gemini-3.5-flash` (its free tier allows 20 requests/day; Flash Lite allows 500). 492 LLM calls for 247
claims (~2.0 per claim, paper: 2.2). eXpath run on exactly the same claims (`compare_expath.py --pool`).

| type | n | PGR | eXpath | PGR strict | eXpath strict |
|---|---|---|---|---|---|
| one-hop | 52 | **0.962** | 0.885 | 0.962 | 0.769 |
| conjunction | 52 | **0.962** | 0.865 | 0.962 | 0.788 |
| existence | 51 | **0.961** | 0.902 | 0.961 | 0.882 |
| multi-hop | 51 | 0.804 | **0.882** | 0.804 | 0.608 |
| negation | 51 | **0.843** | 0.745 | 0.843 | 0.667 |
| **all** | 257 | **0.907** | 0.856 | 0.907 | 0.743 |

FactKG is binary, so the main columns score NOT_ENOUGH_INFO as REFUTED, as the team's `eval_cv.py` does;
"strict" counts it as wrong. PGR gave a verdict for every claim; eXpath said NOT_ENOUGH_INFO for 43.
Gemini-3.5-flash-lite alone: 226/250 = 0.904. Paper (GPT-4o, all 9,041 claims): 0.868; its per-type
numbers: 93.3 / 89.95 / 93.1 / 75.96 / 81.16. The sample is balanced across types, so its "all" weights
the harder types more than the full test set does (eXpath: 0.876 on 1,000 random test claims).
With ~51 claims per type, per-type accuracies are accurate to roughly +/-5-10 points.

Interpreter note: `VERIFY(entity_1 is not None)` (the model's way of asking "did SEARCH find anything?")
is read as an emptiness test, because SEARCH returns `[]`, not `None`, when nothing is found (literal
Python would call every such existence claim SUPPORTED). Lines after the `predicted = ...` line are ignored.

Tests: `python -m pytest tests/test_pgr_core.py` (fake LLM, no key needed).
