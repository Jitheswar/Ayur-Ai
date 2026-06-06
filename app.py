"""Streamlit UI for the Ayurvedic Leaf Detection project.

Run from the project root:

    streamlit run app.py

Three tabs:
  1. Identify a leaf  -- upload an image, get the plant + medicinal properties.
  2. Search by symptom -- natural-language query over the knowledge base (NLP).
  3. Browse herbs      -- explore the full knowledge base.

Tabs 2 and 3 work without a trained model; tab 1 needs models/best_model.pt.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import Config
from src.nlp.knowledge_base import KnowledgeBase
from src.nlp.retriever import PlantRetriever

st.set_page_config(page_title="Ayurvedic Leaf Detector", page_icon="🌿", layout="wide")

cfg = Config.load()


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase.load(cfg.knowledge_base_path)


@st.cache_resource(show_spinner=False)
def get_retriever() -> PlantRetriever:
    return PlantRetriever(get_knowledge_base())


@st.cache_resource(show_spinner=True)
def get_predictor(_ckpt_mtime: float):
    """Load the CNN predictor.

    The checkpoint's mtime is part of the cache key, so retraining the model
    (which rewrites best_model.pt) invalidates the cache and the new weights
    are picked up without restarting the app. The caller only invokes this once
    the checkpoint exists, so we never cache a "no model" state.
    """
    from src.predict import LeafPredictor

    return LeafPredictor(cfg=cfg)


def render_plant(plant, *, score: float | None = None) -> None:
    header = plant.display_name
    if score is not None:
        header += f"  ·  relevance {score:.2f}"
    with st.container(border=True):
        st.markdown(f"### 🌿 {header}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Ayurvedic name:** {plant.ayurvedic_name or '—'}")
            st.markdown(f"**Family:** {plant.family or '—'}")
            st.markdown(f"**Parts used:** {', '.join(plant.parts_used) or '—'}")
        with c2:
            st.markdown(f"**Properties:** {', '.join(plant.medicinal_properties)}")
            st.markdown(f"**Common names:** {', '.join(plant.common_names)}")
        st.markdown(f"**Used for:** {', '.join(plant.uses)}")
        st.info(plant.description)
        if plant.precautions:
            st.warning(f"**Caution:** {plant.precautions}")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🌿 Ayurvedic Leaf Detection & Medicinal Properties")
st.caption(
    "Deep learning identifies the plant from a leaf image; an offline NLP "
    "knowledge base retrieves its medicinal properties — no LLM involved."
)

with st.sidebar:
    st.header("About")
    kb = get_knowledge_base()
    st.metric("Herbs in knowledge base", len(kb))
    model_ready = cfg.checkpoint_path.exists()
    st.metric("Image model", "ready ✅" if model_ready else "not trained ⛔")
    if not model_ready:
        st.caption(
            "Train the classifier to enable image identification:\n\n"
            "`python -m src.train`"
        )
    st.divider()
    st.caption(
        "⚠️ Educational project. The information shown is for learning and "
        "identification only and is **not medical advice**."
    )

tab_identify, tab_search, tab_browse = st.tabs(
    ["🔍 Identify a leaf", "💬 Search by symptom", "📖 Browse herbs"]
)


# --------------------------------------------------------------------------- #
# Tab 1 — Identify a leaf
# --------------------------------------------------------------------------- #
with tab_identify:
    st.subheader("Upload a leaf image")

    if not cfg.checkpoint_path.exists():
        st.warning(
            "No trained model found at `models/best_model.pt`. "
            "Add a dataset under `data/raw/` and run `python -m src.train` first. "
            "Meanwhile, the **Search** and **Browse** tabs work fully."
        )
    else:
        predictor = get_predictor(cfg.checkpoint_path.stat().st_mtime)
        topk = st.slider("How many candidate plants to show", 1, 5, 3)
        upload = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
        if upload is not None:
            col_img, col_res = st.columns([1, 2])
            with col_img:
                st.image(upload, caption="Uploaded leaf", use_container_width=True)

            import tempfile

            # Unique temp file per upload so concurrent users don't clobber
            # each other; removed once the prediction is done.
            suffix = Path(upload.name).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                tf.write(upload.getbuffer())
                tmp = Path(tf.name)
            try:
                preds = predictor.predict(tmp, topk=topk)
            finally:
                tmp.unlink(missing_ok=True)

            with col_res:
                top = preds[0]
                st.success(
                    f"Most likely: **{top.label}**  "
                    f"({top.confidence*100:.1f}% confidence)"
                )
                st.progress(min(1.0, top.confidence))
            for pred in preds:
                if pred.plant is not None:
                    render_plant(pred.plant, score=pred.confidence)
                else:
                    st.error(
                        f"`{pred.label}` ({pred.confidence*100:.1f}%) — "
                        "no knowledge-base entry matched this class name."
                    )


# --------------------------------------------------------------------------- #
# Tab 2 — Search by symptom (NLP retrieval, no LLM)
# --------------------------------------------------------------------------- #
with tab_search:
    st.subheader("Find herbs by symptom or property")
    st.caption(
        "Type a complaint in plain English (e.g. *cough and cold*, "
        "*lower blood sugar*, *skin acne and wounds*). "
        "A TF-IDF model ranks herbs by relevance."
    )
    query = st.text_input("Your query", placeholder="cough and cold relief")
    if query:
        retriever = get_retriever()
        hits = retriever.query(query, top_k=5, min_score=0.01)
        if not hits:
            st.info("No close matches. Try simpler or more common terms.")
        for hit in hits:
            render_plant(hit.plant, score=hit.score)


# --------------------------------------------------------------------------- #
# Tab 3 — Browse the knowledge base
# --------------------------------------------------------------------------- #
with tab_browse:
    st.subheader("Knowledge base")
    names = sorted(p.display_name for p in kb.plants)
    choice = st.selectbox("Pick a herb", names)
    selected = next(p for p in kb.plants if p.display_name == choice)
    render_plant(selected)
