"""End-to-end smoke test of PowerLinkModule on a tiny synthetic KG (no downloads, no GPU, ~1 minute on CPU).

Exercises the real torch + DGL + reference PowerLink code, which the unit tests (tests/test_powerlink_core.py)
deliberately do not. Run inside the kgx-powerlink env, from anywhere, after modules/powerlink/setup_repo.sh:

    python modules/powerlink/scripts/smoke_test.py

It builds a random KG, writes artifacts in exactly the layout train_kge.py produces, trains the KGE for a few
steps, then checks the module's invariants:
  * every explanation step is a real train fact (validates the EID -> relation-id decoding),
  * no NaN/inf anywhere and the responses are strict JSON (allow_nan=False),
  * all three verdicts are reachable and head/tail queries both work,
  * a NaN edge mask degrades to a fallback explanation instead of leaking NaN,
  * the explainer's loss history does not grow between requests.
Exits non-zero on the first failed check.
"""
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "modules" / "powerlink" / "power-link"
sys.path.insert(0, str(ROOT))

N_ENT, N_REL, N_TRAIN, N_EVAL = 150, 4, 800, 50
DATASET = "synth"


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {msg}")
    print(f"  ok: {msg}")


# ----------------------------------------------------------------------------- synthetic data / artifacts

def write_dataset(data_dir: Path, rng: random.Random):
    ents = [f"/m/e{i}" for i in range(N_ENT)]
    rels = [f"/r/rel{i}" for i in range(N_REL)]
    triples = set()
    while len(triples) < N_TRAIN + 2 * N_EVAL:
        h, t = rng.sample(range(N_ENT), 2)
        triples.add((ents[h], rels[rng.randrange(N_REL)], ents[t]))
    triples = sorted(triples)
    rng.shuffle(triples)
    splits = {"train": triples[:N_TRAIN], "valid": triples[N_TRAIN:N_TRAIN + N_EVAL], "test": triples[N_TRAIN + N_EVAL:]}
    folder = data_dir / DATASET
    folder.mkdir(parents=True)
    for name, ts in splits.items():  # CRLF on purpose: the real FB15k-237 files have it
        (folder / f"{name}.txt").write_text("".join("\t".join(t) + "\r\n" for t in ts))
    (folder / "entity2text.txt").write_text("".join(f"{e}\tEntity {i}\n" for i, e in enumerate(ents)))
    return ents, rels, splits


def pretrain_args():
    """The same args train_kge.py passes, parsed by the reference pretrain.py (so config.json has the real keys)."""
    import importlib.util
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))  # pretrain.py imports power_link
    spec = importlib.util.spec_from_file_location("ref_pretrain", REPO / "pretrain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = sys.argv
    sys.argv = ["pretrain.py", "--score_func", "transe", "--opn", "mult", "--gpu", "-1", "--gamma", "9",
                "--hid_drop", "0.2", "--init_dim", "200", "--n_layer", "2", "--encoder", "compgcn", "--data", DATASET]
    try:
        args = module.parse_args()
    finally:
        sys.argv = argv
    args.name = f"compgcn_transe_{DATASET}"
    return args


def write_artifacts(art_dir: Path, ents, rels, train, steps: int = 60) -> None:
    import numpy as np
    import torch
    from modules.powerlink.module import _import_powerlink

    art_dir.mkdir(parents=True)
    (art_dir / "entities.dict").write_text("".join(f"{i}\t{e}\n" for i, e in enumerate(ents)))
    (art_dir / "relations.dict").write_text("".join(f"{i}\t{r}\n" for i, r in enumerate(rels)))
    (art_dir / "config.json").write_text(json.dumps(vars(pretrain_args()), indent=3))

    lib = _import_powerlink(REPO)
    p = lib.load_config(str(art_dir / "config.json"))
    p.embed_dim = p.k_w * p.k_h
    e2i, r2i = {e: i for i, e in enumerate(ents)}, {r: i for i, r in enumerate(rels)}
    tr = np.array([(e2i[h], r2i[r], e2i[t]) for h, r, t in train], dtype=np.int64)
    torch.manual_seed(1)
    g, norm, edge_type = lib.build_graph(len(ents), p, tr, len(rels))
    meta = {"num_ent": len(ents), "num_rels": len(rels), "edge_type": edge_type, "edge_norm": norm}
    model = lib.build_kgc_model_from_config(p, torch.device("cpu"), meta)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    # Like the reference pretrain.py: every triple is a tail query (h, r, ?) AND a head query (t, r + num_rel, ?).
    queries = np.concatenate([tr, np.stack([tr[:, 2], tr[:, 1] + len(rels), tr[:, 0]], axis=1)])
    for _ in range(steps):
        batch = torch.from_numpy(queries[np.random.RandomState(_).randint(0, len(queries), 256)])
        label = torch.zeros(len(batch), len(ents))
        label[torch.arange(len(batch)), batch[:, 2]] = 1
        loss = model.calc_loss(model.score_all(g, batch[:, 0], batch[:, 1]), label)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save({"model": model.state_dict()}, art_dir / "model.pt")


# ----------------------------------------------------------------------------- checks

def is_finite_response(result) -> bool:
    from common.serialization import to_dict
    try:
        json.dumps(to_dict(result), allow_nan=False)  # raises on NaN / inf
        return True
    except ValueError:
        return False


def steps_are_train_facts(explanation, train_set) -> bool:
    return all(s.triple in train_set for s in explanation.steps)


def check_lp(module, train_set, ents, rels) -> None:
    print("link prediction")
    tail_q = module.predict_link(ents[3], rels[1], None, k=5)
    head_q = module.predict_link(None, rels[2], ents[7], k=5)
    for name, res in (("tail", tail_q), ("head", head_q)):
        check(len(res.predictions) == 5, f"{name} query returns k predictions")
        check(is_finite_response(res), f"{name} query is strict JSON (no NaN/inf)")
        explained = [p for p in res.predictions if p.explanation.steps]
        check(bool(explained), f"{name} query has at least one path explanation")
        check(all(steps_are_train_facts(p.explanation, train_set) for p in res.predictions),
              f"{name} query: every explanation step is a train fact")
    check(not any(p.candidate == ents[3] for p in tail_q.predictions), "tail query never predicts the query entity")
    check(len(module._st.explainer.all_loss) == 0, "explainer loss history is cleared after requests")


def check_cv(module, train, test, ents) -> None:
    import modules.powerlink.module as module_mod
    module_mod.parse_claim = lambda _url, claim, _ds: {"triple": claim.split("|")}  # bypass the parser service
    module_mod.get_service = lambda _name: type("Svc", (), {"base_url": "unused"})()
    print("claim verification")
    in_train = module.verify_claim("|".join(train[0]))
    check(in_train.verdict == "SUPPORTED" and in_train.confidence == 1.0, "a train fact is SUPPORTED with confidence 1.0")

    module.cfg.thresholds.support_rank, module.cfg.thresholds.refute_rank = 20, 60
    results = [module.verify_claim("|".join(t)) for t in test]
    verdicts = {r.verdict for r in results}
    check(verdicts == {"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"}, f"all three verdicts reachable (got {sorted(verdicts)})")
    check(all(is_finite_response(r) for r in results), "all verdicts are strict JSON (no NaN/inf)")
    check(all(steps_are_train_facts(r.explanation, set(train)) for r in results),
          "every CV explanation step is a train fact")
    refuted = next(r for r in results if r.verdict == "REFUTED")
    check(refuted.explanation.text.startswith("The model ranks") and "instead favours" in refuted.explanation.text,
          "REFUTED explains the model's preferred alternative")


def check_nan_mask_degrades(module, ents, rels) -> None:
    print("NaN edge mask")
    explainer = module._st.explainer
    real_explain = explainer.explain

    def nan_explain(**kwargs):
        comp_g, paths, masks = real_explain(**kwargs)
        return comp_g, paths, {k: v * float("nan") for k, v in masks.items()}

    explainer.explain = nan_explain
    try:
        res = module.predict_link(ents[3], rels[1], None, k=2)
    finally:
        explainer.explain = real_explain
    check(is_finite_response(res), "NaN mask does not leak NaN into the response")
    check(all("failed" in p.explanation.text and not p.explanation.steps for p in res.predictions),
          "NaN mask yields a fallback explanation that says the explainer failed")


def check_errors(module, ents, rels) -> None:
    from common.errors import ModuleError
    print("error handling")
    for label, call in (("unknown entity", lambda: module.predict_link("/m/nope", rels[0], None)),
                        ("unknown relation", lambda: module.predict_link(ents[0], "/r/nope", None))):
        try:
            call()
        except ModuleError as e:
            check(e.code == "UNKNOWN_ENTITY", f"{label} -> UNKNOWN_ENTITY")
        else:
            raise SystemExit(f"FAIL: {label} did not raise")


def main() -> None:
    if not (REPO / "power_link").is_dir():
        raise SystemExit(f"{REPO} missing. Run modules/powerlink/setup_repo.sh first.")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        os.environ["KGX_DATA_DIR"], os.environ["KGX_ARTIFACTS_DIR"] = str(tmp / "data"), str(tmp / "artifacts")
        ents, rels, splits = write_dataset(tmp / "data", random.Random(0))

        from common.datasets.files import DatasetSpec, build_knowledge_graph
        from modules.powerlink.config import PowerLinkConfig
        from modules.powerlink.module import PowerLinkModule

        write_artifacts(tmp / "artifacts" / "powerlink" / DATASET, ents, rels, splits["train"])
        cfg = PowerLinkConfig(device="cpu", explain_top_n=3)
        cfg.explainer.num_epochs = 10
        module = PowerLinkModule(cfg)
        module.load(build_knowledge_graph(DatasetSpec(name=DATASET)))

        train_set = set(splits["train"])
        check_lp(module, train_set, ents, rels)
        check_cv(module, splits["train"], splits["test"][:25], ents)
        check_nan_mask_degrades(module, ents, rels)
        check_errors(module, ents, rels)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
