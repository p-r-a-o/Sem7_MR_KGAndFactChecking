# AnyBURL+SAFRAN module (rule-based)

Wraps [AnyBURL](https://web.informatik.uni-mannheim.de/AnyBURL/) (Meilicke et al., "Anytime
Bottom-Up Rule Learning for Knowledge Graph Completion", IJCAI'19 and later), which samples paths
from the graph, generalises them into logical rules, and scores each rule by how well it predicts
the training data. Rules are mined by the **official jar**; this module applies them in Python so
that every answer ships with the rule *and* the facts it grounded on.

The [SAFRAN](https://safran.readthedocs.io/en/latest/) side is the **aggregation** choice:
`--aggregation max` is AnyBURL's own (a candidate scores as its most confident rule),
`--aggregation noisy-or` combines all firing rules as `1 - prod(1 - confidence)`. See "Deviations".

## Setup (run from the repo root)
Only the *mining* step needs a JDK; serving uses the standard library alone, so `services.yaml`
starts `api.py` with plain `python` and no environment of its own.
```bash
# skip these two lines if you already have a JDK 11+ on PATH
conda env create -f modules/anyburl_safran/environment.yml   # python 3.10 + openjdk 17
conda activate kgx-anyburl
bash modules/anyburl_safran/setup_jar.sh                     # downloads AnyBURL-23-1x.jar (151 KB)
python scripts/download_data.py --dataset fb15k237           # shared splits -> data/
python modules/anyburl_safran/scripts/mine_rules.py --dataset fb15k237 --seconds 100,1000
python modules/anyburl_safran/api.py                         # port 8102, from services.yaml
```
Artifacts land in `artifacts/anyburl_safran/<dataset>/`:
`rules.txt` (the packaged snapshot, what `load()` reads), `mining.json` (jar, config, rule counts for
the report) and `mining/` (AnyBURL's working directory with the generated properties file and every
snapshot). Claim verification also needs the claim parser service (`services.yaml: claim_parser`).

Optional, before reporting CV numbers on a new dataset:
```bash
python scripts/generate_claims.py --dataset fb15k237 --split valid --n 500
python modules/anyburl_safran/scripts/tune_thresholds.py --dataset fb15k237 --n 500
```

## How it maps to the shared interface
- **`load()`** parses the rule file and builds a `(subject, relation) -> objects` index over the
  **train split only**. No model, no Java at request time.
- **`predict_link`** fires every rule learned for the queried relation from the bound entity,
  aggregates the firing rules per candidate, ranks, and hides answers already in the train graph
  (`--include-known` keeps them). Head queries `(?, r, t)` bind the rule head's *object* instead and
  walk the body the other way, so acyclic rules work in both directions.
- **`verify_claim`** parses the claim to a triple, then:
  1. the triple is in the training graph -> `SUPPORTED`, confidence 1.0;
  2. the relation behaves as a function in training and already states a different answer ->
     `REFUTED`, with the conflicting fact as evidence;
  3. otherwise the verdict follows the claimed entity's rank among rule-derived candidates
     (`support_rank` / `refute_rank`), where "no rule derives it and the rules derived nothing else
     either" is `NOT_ENOUGH_INFO`.
  `score_triple()` returns the raw evidence and `decide()` turns it into a verdict, so threshold
  tuning re-decides cached evidence instead of re-running the reasoning.
- **Explanations** are always `kind="rule"`: `Explanation.rule` is the AnyBURL rule string, the steps
  are the grounded body facts in body order, and the text renders the grounding as a path
  (`Keith Carradine --[sibling]--> David Carradine --[place_of_birth]--> Hollywood`).

## Deviations to confirm with the team
1. **Rule application is reimplemented in Python** (`rules.py` grounding + `core.py` aggregation)
   instead of calling AnyBURL's `Apply`. `Apply` outputs ranked candidates *without* the groundings
   that produced them, and the per-prediction `Explain` entry point exists only in the AnyBURL-22
   build, so there is no supported way to get `HANDOFF.md`'s "rule + its grounding" out of the
   official pipeline. Rule *learning* - the actual method - is still the unmodified official jar.
   Consequence: our ranking is close to but not identical to AnyBURL's published numbers (below).
2. **SAFRAN is approximated, not wrapped.** Real SAFRAN clusters redundant rules
   (non-redundant noisy-or) before combining them; here `--aggregation noisy-or` is plain noisy-or,
   and `dedupe_groundings` (on by default) only drops rules that fired on *identical* evidence.
   Wrapping the real SAFRAN C++ binary would be a separate module.
3. **Zero rules are off** (`ZERO_RULES_ACTIVE = false`, AnyBURL's default): they predict from
   relation frequency alone, so they have no derivation to show.
4. `REFUTED` partly rests on a closed-world reading of functional relations, which `HANDOFF.md`
   section 3 already flags as a source of label noise.

## Measured on FB15k-237

All numbers below were produced by this module, not copied from a paper. Rules: the `rules-1000`
snapshot (1000s of learning, AnyBURL's published setting) with the shipped defaults.
LP: 200 random test triples, both directions (400 queries), k=10.
CV: 500 valid-split claims scored from their **gold triples**, so this measures the reasoning and not
the shared claim parser; `NOT_ENOUGH_INFO` counted as "not supported".

| | FB15k-237 `max` | FB15k-237 `noisy-or` | WN18RR `max` | WN18RR `noisy-or` |
|---|---|---|---|---|
| LP MRR (all) | 0.230 | **0.244** | 0.427 | **0.431** |
| LP Hits@1 | 0.160 | 0.170 | 0.398 | 0.403 |
| LP Hits@10 | 0.420 | **0.450** | 0.485 | 0.487 |
| LP MRR tail / head | 0.308 / 0.152 | 0.322 / 0.166 | 0.447 / 0.406 | 0.454 / 0.407 |
| CV F1 | 0.711 | 0.710 | 0.779 | **0.780** |
| CV accuracy | 0.708 | 0.706 | 0.810 | 0.812 |
| CV precision / recall | 0.695 / 0.729 | 0.692 / 0.729 | 0.954 / 0.657 | 0.960 / 0.657 |
| claims derivable at all | 277/500 | 277/500 | 208/500 | 208/500 |
| rules mined / kept | 4,962,393 / 1,166,696 | | 67,205 / 33,649 | |
| load time / memory | 37s / 1.6GB | | 0.4s / 0.2GB | |
| LP latency / CV latency | 20 ms / 61 ms | 28 ms / 58 ms | 11 ms / 17 ms | 13 ms / 16 ms |

Reference: the AnyBURL papers report ~0.30 MRR on FB15k-237 and ~0.47 on WN18RR with the official
`Apply`. WN18RR lands close; FB15k-237 stays lower, which is consistent with where our reimplemented
applier differs most - FB15k-237's rules are overwhelmingly single-constant acyclic rules
(4.9M of them) that we have to cap, while WN18RR's 11 symmetric/hierarchical relations are carried by
cyclic rules that survive any cap.

### What these numbers say
- **`noisy-or` (SAFRAN-style) beats `max` (AnyBURL's own) on both datasets** once the apply limit is
  high enough - +0.014 MRR and +0.030 Hits@10 on FB15k-237 - at ~40% more time per query. `max` is
  still the default because it is the faithful AnyBURL baseline; consider `--aggregation noisy-or`
  for the demo and report both.
- **CV precision on WN18RR is 0.96 at 0.66 recall.** That is the honest shape of a rule-based
  verifier: it rarely asserts something false, but only 208 of 500 claims are derivable at all, and
  nothing can be proven for the rest. FB15k-237 derives more (277/500) but far less reliably.
- **Head prediction is much weaker than tail prediction** on FB15k-237 (MRR 0.152 vs 0.308) and
  nearly symmetric on WN18RR (0.406 vs 0.447). Mined acyclic rules name a specific *tail* entity in
  their head, so they transfer poorly to `(?, r, t)` queries.
- **Through the full HTTP pipeline including the shared claim parser**, FB15k-237 CV accuracy falls to
  0.577 (F1 0.441) with 33/300 claims unparseable. That gap is the parser, not the reasoning, and is
  the single biggest available CV win.


## Things to know before changing the defaults
1. **Confidence is smoothed.** The rule file's confidence column is the raw
   `correctlyPredicted / predicted`; AnyBURL's `Apply` re-computes it as
   `correctlyPredicted / (predicted + UNSEEN_NEGATIVE_EXAMPLES)` with `UNSEEN_NEGATIVE_EXAMPLES = 5`.
   We do the same (`--unseen-negatives`). It matters a lot: 45k of the mined rules score 1.0 off two
   or three predictions, and without smoothing they outrank every well-supported rule
   (LP MRR 0.118 -> 0.220 on the same rule file when smoothing was added).
2. **Rule files are large.** A 1000s FB15k-237 run produces millions of rules, mostly low-confidence
   acyclic ones. `max_rules_per_relation` keeps the most confident per relation as the file streams
   past; `max_rules` / `saturation_candidates` bound one query. The defaults were picked by measuring
   the quality/latency curve (see `core.ApplyLimits`), not guessed.
3. **Verdict thresholds are tuned on valid claims**, with the caveat that plain accuracy is a bad
   objective here: most positives are not derivable from rules at all, so "refute anything
   unprovable" scores well while answering nothing. `tune_thresholds.py` reports balanced accuracy
   and per-class recall alongside it.
4. **AnyBURL's `Learn` exits non-zero** after writing its last snapshot (seen with AnyBURL-23-1x), so
   `mine_rules.py` treats the snapshot files as the success signal and only warns about the code.
5. The property names written into `config-learn.properties` come from the published quick start; the
   generated file is printed on every run and `--set KEY=VALUE` overrides anything, so a key rejected
   by a future AnyBURL version is easy to fix. (AnyBURL echoes the config it parsed into
   `artifacts/anyburl_safran/<dataset>/mining/rules_log`.)
6. **Entity labels are made readable and unique before display.** Two fixes, both needed for proofs
   to make sense:
   * `common/datasets/wn18rr.py:clean_label` now keeps only the lemma - `data/wn18rr/entity2text.txt`
     stores `"lemma, full dictionary definition"`, so without this every proof carried a whole
     definition. **This touches shared `common/` code** and changes WN18RR labels for every module;
     regenerate WN18RR claims after pulling it (`scripts/generate_claims.py`).
   * `core.disambiguate_labels` appends the entity ID to labels shared by several entities, because
     readable names are not unique: 31% of WN18RR entities share a lemma once the definition is gone
     ("run" covers 33 synsets) and 3% of FB15k-237 names collide ("Washington County" x7). Display
     only - generated claims keep the plain label.
