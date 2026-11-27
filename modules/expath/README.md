# eXpath module (path-based)

Wraps [eXpath](https://github.com/cs-anonymous/expath) (Sun et al., VLDB 2025, "eXpath: Explaining
Knowledge Graph Link Prediction with Ontological Closed Path Rules"). eXpath explains a link
`(h, r, t)` in three stages: **path aggregation** (all paths `h -> t` up to length 3, collapsed to
relation paths), **rule mining** (each relation path is the body of a closed-path rule
`r(X,Y) <- b1(X,A), b2(A,Y)`, scored by support / standard confidence / head coverage; plus
"Uc" rules `r(X, c) <- b(X, c')`), and **critical-fact selection** (noisy-OR over the rules each fact
takes part in).

In this module **every answer is a rule plus a grounding of that rule in the graph**, which can be
re-checked: the facts are looked up, and the rule's SC/HC/support recomputed
(`scripts/check_proof.py`). A score alone is never returned as an answer.

## Setup (run from the repo root)
Serving and mining need only numpy + scipy (no torch), so they run on any laptop:
```bash
pip install -r modules/expath/requirements.txt
# FB15k-237 (link prediction)
python scripts/download_data.py --dataset fb15k237
python modules/expath/scripts/mine_rules.py --dataset fb15k237         # ~25 min on a laptop CPU
# FactKG + DBpedia (claim verification). Download the FactKG Google Drive folder
# (https://github.com/jiho283/FactKG) into data/factkg_raw/: factkg.zip (unzip it) and
# dbpedia_2015_undirected_light.pickle (1.5 GB). The 5.5 GB full graph is not needed.
python scripts/prepare_factkg.py                                       # ~15 min, needs ~10 GB RAM once
python modules/expath/scripts/build_lexicon.py --dataset factkg        # seconds
python modules/expath/api.py                     # port 8103 (services.yaml); add --enrich for node addition
```
Optional: the KGE base model eXpath explains (needs a GPU; e.g. Colab), see "Base model" below.

Artifacts land in `artifacts/expath/<dataset>/`: `rules.jsonl` (mined rules with their metrics),
`mining.json` (settings, counts, timings), `lexicon.json` (FactKG), `kge/` (optional exported
embeddings), `additions.jsonl` (facts added by node addition, with source and date).

## How it maps to the shared interface
- **`load()`** indexes the TRAIN split only (CSR adjacency with inverse edges, one sparse matrix
  per relation) and reads the artifacts. No training at request time.
- **`predict_link(h, r, ?)`** (head queries become `(t, r^-1, ?)`; mined rules include inverses).
  KGE mode is used automatically when `artifacts/expath/<ds>/kge/` exists (FB15k-237: yes, see Measured):
  - *rules mode* (default when no KGE is exported): apply every mined rule for `r` forward from the
    query entity, aggregate per candidate (noisy-OR, eXpath's `calculateConfidence`; `--aggregation max`
    available), hide answers already in train. Each prediction's explanation = best rule +
    one grounded path, further firing rules in the text.
  - *KGE mode* (when `artifacts/expath/<ds>/kge/` exists): the ComplEx model ranks all entities, and
    eXpath explains each of the top `kge_candidates` with `explain_triple`. Candidates eXpath cannot
    prove are **dropped** (`--allow-unproven` keeps them); `score` is the KGE probability,
    `explanation.confidence` the eXpath rule confidence.
- **`verify_claim`** uses `claim_mode` from config (per dataset, not code branches):
  - `triple` (FB15k-237 / WN18RR): shared claim parser -> `(h, r, t)`. Stated fact -> SUPPORTED (1.0).
    eXpath confidence >= `support` -> SUPPORTED with the rule proof. Functional relation already
    pointing elsewhere, or a better-proven alternative answer -> REFUTED with that evidence.
    Otherwise NOT_ENOUGH_INFO (the explanation lists what was searched and which rules were weak).
  - `multi_entity` (FactKG): see `factcheck.py`. Entities = FactKG's given `Entity_set` (the dataset's
    standard input, present in all splits), else a label linker for free text. Relation relevance =
    lexical match with the relation name, or `P(relation | word)` learned from the gold evidence
    paths of FactKG **train** claims. SUPPORTED when all claim entities are connected by
    claim-relevant paths (or by a link eXpath *infers* with a rule - KG completion proving veracity);
    REFUTED when an entity cannot be reached but a claim-relevant relation points elsewhere
    (the conflicting fact is the proof); a negated claim swaps the two; else NOT_ENOUGH_INFO.

### Node addition (professor's request)
With `--enrich`, when a claim entity is missing from the graph or claim entities cannot be
connected, `enrich.py` queries the public DBpedia SPARQL endpoint for that entity's
`dbpedia.org/ontology/*` facts (wiki-link noise excluded), restricted to relations the graph already
has. New entities become **new nodes**; new edges go into an overlay with provenance, are cited in the
proof as `[source: DBpedia SPARQL ..., retrieved <date>]`, and are appended to `additions.jsonl`.
Mined rule statistics stay those of the frozen graph. It is off by default so evaluation is
reproducible; 1,005 of the 54,349 FactKG claim entities are absent from DBpedia-light, which is where
it matters most.

## Checking a proof
```bash
curl -s localhost:8103/predict_link -d '{"head":"/m/0f8l9c","relation":"...","tail":null,"k":5}' > out.json
python modules/expath/scripts/check_proof.py --dataset fb15k237 out.json
```
For each explanation: every cited fact is looked up in the graph (externally sourced facts are flagged
as such), and the cited rule is parsed and its SC / HC / support recomputed and compared with the text.

## Base model (optional, GPU) - e.g. on Kaggle
1. New Kaggle notebook -> Settings: Accelerator **GPU**, Internet **On**.
2. Paste `scripts/kaggle_train_complex.py` into one cell and run it (it clones eXpath, runs its unmodified
   `train.py` for ComplEx on FB15k-237 and prints test MRR, then exports). Download
   `/kaggle/working/kge_fb15k237.zip` from the notebook's Output panel.
3. Here: unzip into `artifacts/expath/fb15k237/kge/` and restart `api.py`; the load line then says
   `KGE loaded` and link prediction switches to KGE mode (`--no-kge` switches back).
Checked locally with a fake export in the same layout: all 14,541 entities / 237 relations map (eXpath's
FB15k-237 files are byte-identical to ours), and forward and inverse scores match eXpath's ComplEx formula.
In KGE mode a query explains up to `kge_candidates` (100) candidates, so an untrained or poor model
(few provable candidates) can take ~30 s per query and return fewer than k answers.

Original route (needs this repo on the GPU machine):
eXpath is an explainer of an embedding model. `scripts/train_kge.py` runs the **unmodified reference
`train.py`** with eXpath's own `config.yaml` hyper-parameters (ComplEx), then exports the embeddings
to `.npy`; `kge.py` scores them with numpy using the same formula as eXpath's ComplEx. Run it in
`environment-kge.yml` (torch + CUDA), e.g. on Colab, after `bash modules/expath/setup_repo.sh`, and copy
`artifacts/expath/fb15k237/kge/` back. Done for FB15k-237 via the Kaggle route above (see Measured).

## Measured (all produced by this code on a laptop CPU: 8 cores, 16 GB RAM, no GPU)

**Link prediction, FB15k-237** - rules mode (no KGE), 200 random test triples x both directions = 400
queries, k=10, train-known answers hidden; same protocol and seed as the AnyBURL+SAFRAN module.

| | eXpath `noisy-or` (default) | eXpath `max-plus` | AnyBURL+SAFRAN `noisy-or` (teammate) |
|---|---|---|---|
| MRR (all) | **0.233** | 0.221 | 0.244 |
| Hits@1 / @3 / @10 | 0.160 / 0.273 / 0.418 | 0.155 / 0.253 / 0.405 | 0.170 / - / 0.450 |
| MRR tail / head | 0.311 / 0.155 | 0.310 / 0.132 | 0.322 / 0.166 |
| rules | 17,792 CP + 183,021 Uc | | 1,166,696 |
| latency | 0.23 s/query incl. proofs | | |

Mining: 24 min (CP) + 4 s (Uc). Proof check: all 474 predictions of 100 random test queries
(307 CP-rule proofs, 167 Uc-rule proofs) re-verified by `check_explanation` - every cited fact is in the
graph and every rule's SC/HC/support recomputes to the stated values.

**Claim verification, FactKG** (`NOT_ENOUGH_INFO` scored as REFUTED, since FactKG labels are binary;
uses FactKG's given entity sets). Random 1,000-claim samples (`--shuffle-seed`), not the full splits.

| | valid (1,000, seed 1) | test (1,000, seed 0) |
|---|---|---|
| accuracy | 0.833 | **0.876** |
| F1 (SUPPORTED) / precision / recall | 0.811 / 0.859 / 0.769 | 0.867 / 0.898 / 0.838 |
| one-hop / conjunction / existence / multi-hop / negation acc. | 0.80 / 0.81 / 0.89 / 0.86 / 0.71 | 0.83 / 0.88 / 0.94 / 0.88 / 0.75 |
| NOT_ENOUGH_INFO rate | 0.21 | 0.17 |
| latency median / p95 / max | 0.04 / 0.91 / 6.5 s | 0.04 / 0.54 / 4.2 s |

Every one of the 2,000 verdicts carries proof steps; 100 random test verdicts were re-checked with
`check_explanation` (all facts present). Note on test use: the test sample was first scored *before* the
closed-world existence rule was added (accuracy 0.861, F1 0.848, negation 0.63). The change was then
validated on valid (0.818 -> 0.833) and test scored once more; both runs are reported here.
The FactKG paper's best KG-evidence baseline (GEAR) reports ~0.78 test accuracy on the full test split -
a different sample and protocol, so this is context, not a head-to-head comparison.
Rule-inferred links (KG completion as proof) are rare on FactKG: 4 / 1,000 valid and 1 / 1,000 test
verdicts use one (e.g. `dateOfBirth(Y, X) <- birthDate(Y, X)`, SC 0.766); FactKG claims are mostly about
stored facts. No rules were mined on the 2.4M-entity FactKG graph (only computed on demand, capped).

**Node addition (DBpedia SPARQL), FactKG valid:** 108 claims name an entity absent from the graph.
Without `--enrich` all 108 are NOT_ENOUGH_INFO. With it, DBpedia returned ontology facts for 24 of
those entities (126 facts, all new nodes logged in `additions.jsonl` with source and date); those 24
claims went from 0 to 24 decided, accuracy 0.667 -> 0.917. The other 84 entities have no DBpedia
ontology facts at all, so NOT_ENOUGH_INFO remains the honest answer.

**Claim verification, FB15k-237** (triple mode via the shared claim parser, synthetic claims):
valid 300: support threshold 0.3 / 0.5 / 0.7 / 0.85 -> F1 0.643 / **0.679** / 0.673 / 0.669 (0.5 kept);
test 600: accuracy 0.632, F1 0.678 (precision 0.602, recall 0.777).

**Link prediction with the KGE base model (KGE mode), FB15k-237** - ComplEx trained on Kaggle with
`scripts/kaggle_train_complex.py` (eXpath's unmodified `train.py` and settings; Kaggle reported test MRR
0.3653 / H@1 0.2707 / H@10 0.5575 / MR 148.7). Our numpy scorer reproduces those four numbers exactly on all
20,466 test triples (standard filtered setting), so export and ID mapping are verified.
Same 400-query protocol as the table above:

| | KGE alone (no proofs) | eXpath KGE mode (every answer proven) | eXpath rules mode |
|---|---|---|---|
| MRR (all) | 0.260 | **0.259** | 0.233 |
| Hits@1 / @3 / @10 | 0.172 / 0.307 / 0.485 | 0.172 / 0.305 / 0.482 | 0.160 / 0.273 / 0.418 |
| MRR tail / head | | 0.344 / 0.175 | 0.311 / 0.155 |
| latency | 4 ms/query | 1.0 s/query incl. proofs | 0.23 s/query |

Requiring a proof costs almost nothing (MRR 0.259 vs 0.260): unprovable candidates are rare (4 of 100
queries dropped any). All 481 predictions of 100 random queries passed
`check_explanation`. (The MRR 0.365 above is the full filtered ranking; this table hides only train-known
answers and counts top-10 only, like every module's `eval_lp.py` numbers.)

**Run `eval_lp.py` against a server started without `--show-known`.** `services.yaml` starts eXpath with
`--show-known` so the app lists the graph's stored answers first (marked "already in the knowledge graph").
`eval_lp.py` scores list positions, so with the flag the stored answers would take the top ranks.

Tests: `python -m pytest tests/test_expath_core.py` (toy graph with hand-counted rule metrics, proof
re-check, LP both directions, multi-entity support / refute / negation / closed-world existence).

## Known weaknesses (seen in the Streamlit demo)
- **A relevant hop can carry an irrelevant one.** A path's relevance is the mean over its relations, so
  "Barack Obama was born in Chicago" (false) is SUPPORTED via "Obama --successor--> Kwame Raoul
  --birthPlace--> Chicago". `--hop-relevance 0.05` rejects such paths (that claim becomes REFUTED with
  "Obama --birthPlace--> Honolulu" as proof) but lowers FactKG valid accuracy 0.832 -> 0.799, because
  many genuine FactKG hops are not signalled by any claim word the lexicon knows. Default: off.
- The "why relevant" note comes from word/relation co-occurrence in FactKG train claims and can be odd
  (e.g. "weigh" -> placeOfBirth, from conjunction claims mentioning both); verdicts and facts are unaffected.
- Free-text claims (not in FactKG) are linked by label match: a match must start with a capitalised word.
- The app's "sample claim" list shows the first 50 FactKG test claims, which are all existence claims
  (the file is sorted by type); type claims to try the other kinds.

## Deviations from the reference (confirm with the team)
1. **eXpath's stages are ported, not imported.** `process.py` parses `argv` and reads `out/...json`
   at import time, so it cannot be imported. Ported with the same formulas: path search
   (`find_paths_bfs` -> `graph.enumerate_paths`, meet-in-the-middle with hub/sampling caps), CP metrics
   (`calculate_rule_metrics_with_matrix` -> `rules.cp_metrics`, same binarised sparse products,
   computed in row blocks), Uc metrics (`calculate_Uc_metrics`), rule score `SC*supp/(supp+10)`, fact
   importance and noisy-OR. Thresholds are eXpath's (minSC = minHC = 0.01).
2. **No Kelpie post-training relevance (Rh, Rt) and no Ud rules.** eXpath weights facts by how much
   retraining the KGE without them lowers the prediction score. That costs several GPU retrainings per
   prediction and is impossible here. We use eXpath's own fallback for Rh = Rt = 0 (plain average of
   head/tail proportions). Ud rules depend only on that relevance, so they are omitted.
3. **Rules are also used to predict, not only to explain.** eXpath mines rules per explained
   prediction. We additionally mine them once per relation (`mine_rules.py`, sampled) so LP works
   without a GPU-trained KGE, and so a proof exists for every returned answer.
4. **FactKG claims bypass the shared claim parser** (multi-entity claims don't fit one triple) and
   use FactKG's given entity set. `common/` changes made for FactKG are listed in the repo README.
5. Verdict thresholds in `config.VerdictConfig`: triple-mode `support` chosen on FB15k-237 valid claims;
   FactKG thresholds are the untuned defaults (only the closed-world existence rule was added after a
   valid-split check, see Measured).
6. **Closed-world existence.** A FactKG existence claim ("X had a successor") about an entity the graph
   describes, with no claim-relevant relation among its facts, is REFUTED; the proof lists X's facts.
   This is FactKG's own reading of existence claims, but it is a closed-world assumption (config flag).
8. Results do not depend on query history: rule metrics computed on demand are cached, but every
   query is charged its budget for each on-demand rule it considers, cached or not (an earlier version
   let a warm cache change verdicts; the FB15k-237 numbers above were re-measured after the fix).
7. On-demand rule metrics are skipped when a body has more than 5M groundings (`path_instances`,
   counted exactly with sparse mat-vec products); without this cap single DBpedia claims took > 10 min.
