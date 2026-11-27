"""PowerLinkModule: wraps the PowerLink reference code behind the shared ReasoningModule interface.

Needs the `kgx-powerlink` environment (python 3.10, torch 1.12, dgl 0.9) and the cloned reference repo
(modules/powerlink/setup_repo.sh). Pure logic lives in core.py so it can be tested without torch.

Design notes
  * LP: score (src, rel) against all entities with the KGE model, drop already-known answers, then
    run PowerLink's explainer on each top-n candidate to obtain explanation paths.
  * Head queries (?, r, t) are answered as tail queries (t, r^-1, ?) using PowerLink's inverse
    relations (id r + num_rel), as in the reference evaluation; paths are reversed for display.
  * CV: parse claim -> triple; verdict from the *filtered rank* of the claimed tail among all
    tails for (head, relation). Thresholds are config (see config.VerdictThresholds), untuned.
  * The message-passing graph is built from TRAIN triples only, so test facts are never available
    to the model or the explainer.
"""
import dataclasses
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import dgl
import numpy as np
import torch

from common.client import parse_claim
from common.datasets.base import KnowledgeGraph
from common.errors import ModuleError
from common.schema import (CVResult, Explanation, ExplanationStep, LinkPrediction, LPResult)
from common.services import get_service
from modules.powerlink import core
from modules.powerlink.config import PowerLinkConfig

ARTIFACT_FILES = ("config.json", "model.pt", "entities.dict", "relations.dict")


def _import_powerlink(repo_dir: Path) -> SimpleNamespace:
    if not (repo_dir / "power_link").is_dir():
        raise ModuleError("INTERNAL", f"PowerLink repo not found at {repo_dir}. Run modules/powerlink/setup_repo.sh.")
    if str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))  # appended: our own packages keep priority
    from power_link.explainer import PowerLinkExplainer
    from power_link.kge import build_kgc_model_from_config, load_config
    from power_link.kge.load_kg import build_graph
    from power_link.utils import set_seed
    return SimpleNamespace(PowerLinkExplainer=PowerLinkExplainer, build_graph=build_graph,
                           build_kgc_model_from_config=build_kgc_model_from_config,
                           load_config=load_config, set_seed=set_seed)


def read_dict(path: Path) -> list[str]:
    """Read PowerLink's '<index>\\t<name>' dict file into a list indexed by id."""
    pairs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 2:
                pairs[int(parts[0])] = parts[1]
    if sorted(pairs) != list(range(len(pairs))):
        raise ModuleError("INTERNAL", f"{path} ids are not contiguous from 0")
    return [pairs[i] for i in range(len(pairs))]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


@dataclass
class _State:
    kg: KnowledgeGraph
    vocab: core.Vocab
    ent2idx: dict[str, int]
    rel2idx: dict[str, int]
    num_rel: int
    model: object
    explainer: object
    mp_g: object
    edge_type: list[int]                                # per mp_g edge id -> relation id (incl. inverse)
    known: dict[tuple[int, int], set[int]]              # (src, rel_id) -> known answers in train
    train_set: set[tuple[int, int, int]]


class PowerLinkModule:
    name = "PowerLink"

    def __init__(self, config: Optional[PowerLinkConfig] = None):
        self.cfg = config or PowerLinkConfig()
        self._st: Optional[_State] = None
        self._lib: Optional[SimpleNamespace] = None

    # ------------------------------------------------------------------ loading

    def load(self, kg: KnowledgeGraph) -> None:
        art = self.cfg.artifact_root / "powerlink" / kg.name
        missing = [f for f in ARTIFACT_FILES if not (art / f).exists()]
        if missing:
            raise ModuleError("INTERNAL", f"Missing PowerLink artifacts {missing} in {art}. "
                                          f"Run modules/powerlink/scripts/train_kge.py --dataset {kg.name}.")
        lib = self._lib = self._lib or _import_powerlink(self.cfg.repo_dir)
        lib.set_seed(self.cfg.seed)
        device = torch.device(self.cfg.device)

        ent_ids, rel_names = read_dict(art / "entities.dict"), read_dict(art / "relations.dict")
        ent2idx = {e: i for i, e in enumerate(ent_ids)}
        rel2idx = {r: i for i, r in enumerate(rel_names)}
        num_ent, num_rel = len(ent_ids), len(rel_names)
        try:
            train = np.array([(ent2idx[h], rel2idx[r], ent2idx[t]) for h, r, t in kg.train], dtype=np.int64)
        except KeyError as e:
            raise ModuleError("INTERNAL", f"{e} from the dataset is not in the PowerLink dicts; "
                                          "data/ and artifacts/ come from different dataset versions.")

        p = lib.load_config(str(art / "config.json"))
        if getattr(p, "embed_dim", None) is None:
            p.embed_dim = p.k_w * p.k_h
        g, norm, edge_type = lib.build_graph(num_ent, p, train, num_rel)
        mp_g = g.to(device)
        meta = {"num_ent": num_ent, "num_rels": num_rel,
                "edge_type": edge_type.to(device), "edge_norm": norm.to(device)}

        model = lib.build_kgc_model_from_config(p, device, meta)
        model.src_ntype, model.tgt_ntype = "_N", "_N"
        state = torch.load(art / "model.pt", map_location=device)["model"]
        state.pop("bias", None)
        model.load_state_dict(state)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        ex = self.cfg.explainer
        explainer = lib.PowerLinkExplainer(model, lr=ex.lr, num_epochs=ex.num_epochs, alpha=ex.alpha,
                                           beta=ex.beta, log=False, embed_dim=p.embed_dim,
                                           parameterizer_hidden_dim=ex.parameterizer_hidden_dim).to(device)

        known: dict[tuple[int, int], set[int]] = {}
        for h, r, t in train.tolist():
            known.setdefault((h, r), set()).add(t)
            known.setdefault((t, r + num_rel), set()).add(h)

        self._st = _State(kg=kg, vocab=core.Vocab(ent_ids, rel_names, kg.entity_labels),
                          ent2idx=ent2idx, rel2idx=rel2idx, num_rel=num_rel, model=model,
                          explainer=explainer, mp_g=mp_g, edge_type=edge_type.tolist(), known=known,
                          train_set={tuple(x) for x in train.tolist()})

    # ------------------------------------------------------------------ helpers

    def _need(self) -> _State:
        if self._st is None:
            raise ModuleError("NOT_LOADED", "PowerLink: call load() first.")
        return self._st

    def _ent(self, entity: str) -> int:
        try:
            return self._need().ent2idx[entity]
        except KeyError:
            raise ModuleError("UNKNOWN_ENTITY", f"Unknown entity: {entity!r}")

    def _rel(self, relation: str) -> int:
        try:
            return self._need().rel2idx[relation]
        except KeyError:
            raise ModuleError("UNKNOWN_ENTITY", f"Unknown relation: {relation!r}")

    def _tensor(self, value: int) -> torch.Tensor:
        return torch.tensor(value, device=torch.device(self.cfg.device))

    def _scores(self, src: int, rel_id: int) -> list[float]:
        """Raw logits of (src, rel_id, e) for every entity e (sigmoid-monotonic, as in the reference)."""
        st = self._need()
        with torch.no_grad():
            out = st.model.score_all(st.mp_g, torch.tensor([src], device=torch.device(self.cfg.device)),
                                     torch.tensor([rel_id], device=torch.device(self.cfg.device)))
        return out[0].detach().cpu().tolist()

    def _paths(self, src: int, rel_id: int, cand: int) -> Optional[list[list[core.Hop]]]:
        """Run PowerLink's explainer for the link (src, rel_id, cand); decode its paths. None on failure."""
        st, ex = self._need(), self.cfg.explainer
        try:
            comp_g, paths, mask_dict = st.explainer.explain(
                src_nid=self._tensor(src), tgt_nid=self._tensor(cand), rel_id=self._tensor(rel_id),
                ghetero=st.mp_g, num_hops=ex.num_hops, prune_max_degree=ex.prune_max_degree, k_core=ex.k_core,
                num_paths=ex.num_paths, max_path_length=ex.max_path_length, prune_graph=True,
                without_path_loss=False, without_mi=False, return_mask=True,
                regularisation_weight=ex.regularisation_weight, power_order=ex.power_order,
                comp_g_size_limit=ex.comp_g_size_limit, pagelink=False, combination_method=ex.combination_method)
        finally:
            st.explainer.all_loss.clear()  # explainer appends per-epoch losses forever; a long-lived service must not leak
        if comp_g is None or not paths:
            return None
        mask = next(iter(mask_dict.values())).detach().cpu()
        if torch.isnan(mask).any():
            # Same check as the reference run_powerlink.py. Happens when sigmoid(score) underflows to 0
            # (very implausible link): -log(0) = inf, so the mask gradients are NaN.
            raise RuntimeError("edge mask diverged to NaN (the link's score is numerically zero)")
        comp_g = comp_g.to("cpu")
        src_l, dst_l = (x.tolist() for x in comp_g.edges())
        rels = [st.edge_type[e] for e in comp_g.edata[dgl.EID].tolist()]
        weights = torch.sigmoid(mask).tolist()
        if len(weights) != len(src_l):
            weights = [1.0] * len(src_l)
        pairs = core.pair_index(src_l, dst_l, rels, weights)
        node_ids = comp_g.ndata[dgl.NID].tolist()
        return [core.decode_path([(hop[1], hop[2]) for hop in path], pairs, node_ids) for path in paths]

    def _explain(self, src: int, rel_id: int, cand: int, reverse: bool, intro: str) -> Explanation:
        st = self._need()
        try:
            hop_paths = self._paths(src, rel_id, cand)
        except RuntimeError as e:  # e.g. CUDA OOM or numerical failure inside the explainer
            return core.fallback_explanation(f"{intro} The path explainer failed ({e}).")
        if hop_paths is None:
            return core.fallback_explanation(
                f"{intro} No path explanation: the computation graph around this pair exceeds "
                f"comp_g_size_limit={self.cfg.explainer.comp_g_size_limit} or no paths were found.")
        if reverse:
            hop_paths = [core.reverse_hops(p, st.num_rel) for p in hop_paths]
        return (core.path_explanation(hop_paths, st.num_rel, st.vocab, intro, self.cfg.explainer.paths_returned)
                or core.fallback_explanation(f"{intro} The explainer returned no usable path."))

    # ------------------------------------------------------------------ interface

    def predict_link(self, head: Optional[str], relation: str, tail: Optional[str], k: int = 10) -> LPResult:
        st = self._need()
        rel = self._rel(relation)
        reverse = head is None
        src = self._ent(tail if reverse else head)
        rel_id = rel + st.num_rel if reverse else rel
        scores = self._scores(src, rel_id)
        exclude = {src} | (st.known.get((src, rel_id), set()) if self.cfg.exclude_known else set())
        predictions = []
        for n, cand in enumerate(core.top_k(scores, k, exclude)):
            label = st.vocab.label(cand)
            if n < self.cfg.explain_top_n:
                intro = (f"Predicted {label} as the head of (?, {relation}, {st.vocab.label(src)})." if reverse
                         else f"Predicted {label} as the tail of ({st.vocab.label(src)}, {relation}, ?).")
                explanation = self._explain(src, rel_id, cand, reverse, intro)
            else:
                explanation = core.fallback_explanation(
                    f"Ranked #{n + 1} by the KGE model; path explanation skipped (explain_top_n="
                    f"{self.cfg.explain_top_n}).")
            predictions.append(LinkPrediction(st.vocab.ent_ids[cand], _sigmoid(scores[cand]), explanation))
        return LPResult(query=(head, relation, tail), predictions=predictions)

    def verify_claim(self, claim: str) -> CVResult:
        st = self._need()
        parser_url = self.cfg.parser_url or get_service("claim_parser").base_url
        h_id, r_name, t_id = parse_claim(parser_url, claim, st.kg.name)["triple"]
        h, r, t = self._ent(h_id), self._rel(r_name), self._ent(t_id)
        triple = (h_id, r_name, t_id)
        lab = st.vocab.label

        if (h, r, t) in st.train_set:
            step = ExplanationStep(triple=triple, note="Fact present in the knowledge graph.")
            return CVResult(claim, triple, "SUPPORTED", 1.0, Explanation(
                kind="path", steps=[step], confidence=1.0,
                text=f"({lab(h)}, {r_name}, {lab(t)}) is stated directly in the knowledge graph."))

        scores = self._scores(h, r)
        rank = core.filtered_rank(scores, t, st.known.get((h, r), set()))
        th = self.cfg.thresholds
        verdict = core.decide_verdict(rank, th.support_rank, th.refute_rank)
        prob = _sigmoid(scores[t])
        prefix = (f"The model ranks {lab(t)} #{rank} of {len(st.vocab.ent_ids)} candidates for "
                  f"({lab(h)}, {r_name}, ?).")

        if verdict == "REFUTED":
            alt = core.top_k(scores, 1, {h} | st.known.get((h, r), set()))[0]
            expl = self._explain(h, r, alt, False,
                                 f"The graph instead favours {lab(alt)} for ({lab(h)}, {r_name}, ?).")
            confidence: Optional[float] = 1.0 - prob
        else:
            expl = self._explain(h, r, t, False, f"Claimed link: ({lab(h)}, {r_name}, {lab(t)}).")
            confidence = prob if verdict == "SUPPORTED" else None
        return CVResult(claim, triple, verdict, confidence, dataclasses.replace(expl, text=f"{prefix} {expl.text}"))
