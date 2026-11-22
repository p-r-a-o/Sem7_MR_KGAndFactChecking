"""Streamlit UI. Talks to modules only through the registry (HTTP proxies from services.yaml).

Run: streamlit run app/streamlit_app.py
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from common.datasets.base import KnowledgeGraph, get_claim_source, load_graph  # noqa: E402
from common.errors import ModuleError  # noqa: E402
from common.registry import build_registry  # noqa: E402
from common.schema import Explanation  # noqa: E402

st.set_page_config(page_title="Explainable KG reasoning", layout="wide")


@st.cache_resource
def get_registry():
    return build_registry()


@st.cache_resource
def get_kg(name: str) -> KnowledgeGraph:
    return load_graph(name)


@st.cache_data
def sample_claims(dataset: str, n: int = 50):
    try:
        return list(itertools.islice(get_claim_source(dataset).claims("test"), n))
    except FileNotFoundError:
        return []


def ensure_loaded(module, kg: KnowledgeGraph) -> None:
    if module.health().get("loaded") != kg.name:
        with st.spinner(f"Loading {kg.name} into {module.name}..."):
            module.load(kg)


def label(kg: KnowledgeGraph, entity_id: str) -> str:
    return kg.entity_labels.get(entity_id, entity_id)


def show_explanation(kg: KnowledgeGraph, ex: Explanation) -> None:
    st.markdown(ex.text)
    if ex.rule:
        st.code(ex.rule)
    if ex.confidence is not None:
        st.caption(f"Explanation confidence: {ex.confidence:.2f}")
    for i, step in enumerate(ex.steps, 1):
        if step.triple:
            h, r, t = step.triple
            st.markdown(f"{i}. **{label(kg, h)}** — `{r}` → **{label(kg, t)}**  \n   {step.note}")
        else:
            st.markdown(f"{i}. {step.note}")


def lp_tab(module, kg: KnowledgeGraph) -> None:
    c1, c2 = st.columns(2)
    relation = c1.selectbox("Relation", kg.relations)
    side = c2.radio("Predict", ["tail  (head, r, ?)", "head  (?, r, tail)"], horizontal=True)
    query = st.text_input("Search entity name (or paste an ID)")
    ids = [e for e, name in kg.entity_labels.items()
           if query.lower() in name.lower() or query == e] if query else []
    entity = st.selectbox("Entity", ids[:100], format_func=lambda e: f"{label(kg, e)}  [{e}]") if ids else None
    k = st.slider("Top-k", 1, 20, 5)
    if st.button("Predict", disabled=entity is None, key="lp_go"):
        head, tail = (entity, None) if side.startswith("tail") else (None, entity)
        try:
            with st.spinner("Predicting and explaining (path explanations can take a while)..."):
                res = module.predict_link(head, relation, tail, k)
        except ModuleError as e:
            return st.error(f"{e.code}: {e.message}")
        for i, p in enumerate(res.predictions, 1):
            with st.expander(f"#{i}  {label(kg, p.candidate)}   (score {p.score:.3f})", expanded=i == 1):
                show_explanation(kg, p.explanation)


def cv_tab(module, kg: KnowledgeGraph) -> None:
    samples = sample_claims(kg.name)
    choice = st.selectbox("Sample claim (test split)", [None] + samples,
                          format_func=lambda c: "— write my own —" if c is None else f"[{c.label}] {c.claim}")
    claim = st.text_area("Claim", value=choice.claim if choice else "", height=80)
    if st.button("Verify", disabled=not claim.strip(), key="cv_go"):
        try:
            with st.spinner("Verifying..."):
                res = module.verify_claim(claim)
        except ModuleError as e:
            return st.error(f"{e.code}: {e.message}")
        color = {"SUPPORTED": "green", "REFUTED": "red", "NOT_ENOUGH_INFO": "orange"}[res.verdict]
        conf = f" (confidence {res.confidence:.2f})" if res.confidence is not None else ""
        st.markdown(f"### :{color}[{res.verdict}]{conf}")
        if res.triple:
            h, r, t = res.triple
            st.caption(f"Parsed as: {label(kg, h)} — {r} → {label(kg, t)}")
        if choice:
            st.caption(f"Gold label: {choice.label}")
        show_explanation(kg, res.explanation)


def main() -> None:
    st.title("Explainable knowledge-graph reasoning")
    registry = get_registry()
    if not registry:
        return st.error("No modules found in services.yaml.")
    module = registry[st.sidebar.selectbox("Module", list(registry))]
    kg = get_kg(st.sidebar.selectbox("Dataset", ["fb15k237", "wn18rr"]))
    try:
        ensure_loaded(module, kg)
    except ModuleError as e:
        return st.error(f"{module.name} is not ready: {e.code}: {e.message}")
    tab_lp, tab_cv = st.tabs(["Link prediction", "Claim verification"])
    with tab_lp:
        lp_tab(module, kg)
    with tab_cv:
        cv_tab(module, kg)


main()
