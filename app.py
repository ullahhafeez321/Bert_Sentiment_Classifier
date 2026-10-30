import os
import re
import html
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = os.getenv(
    "HF_MODEL_ID",
    "ullahhafeez321/bert-sentiment-classifier"
)

st.set_page_config(
    page_title="BERT Sentiment Studio",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { color-scheme: dark; }
.stApp {
    background:
      radial-gradient(ellipse at 12% 0%, rgba(76, 110, 245, .18), transparent 34%),
      radial-gradient(ellipse at 90% 8%, rgba(20, 184, 166, .12), transparent 30%),
      #0b1020;
    color: #edf2ff;
    font-family: 'DM Sans', sans-serif;
}
.block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem; }
h1,h2,h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; }
.hero {
    padding: 2rem 2.1rem; border: 1px solid rgba(148,163,184,.2);
    border-radius: 24px; background: linear-gradient(135deg, rgba(30,41,75,.88), rgba(15,23,42,.88));
    box-shadow: 0 20px 70px rgba(0,0,0,.22); margin-bottom: 1.3rem;
}
.eyebrow { color:#8fb4ff; font-size:.78rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
.hero h1 { font-size: clamp(2.1rem, 5vw, 3.5rem); margin:.35rem 0 .6rem; color:#f8fbff; }
.hero p { color:#b8c4dc; font-size:1.04rem; max-width:720px; margin:0; line-height:1.65; }
.pill {
    display:inline-block; padding:.35rem .7rem; border:1px solid rgba(148,163,184,.25);
    border-radius:999px; color:#cbd5e1; font-size:.78rem; margin:.8rem .35rem 0 0;
    background:rgba(15,23,42,.6);
}
.panel {
    border:1px solid rgba(148,163,184,.18); border-radius:20px; padding:1.25rem 1.35rem;
    background:rgba(15,23,42,.72); box-shadow:0 12px 35px rgba(0,0,0,.12);
}
.metric {
    border:1px solid rgba(148,163,184,.18); border-radius:16px; padding:1rem 1.1rem;
    background:rgba(30,41,59,.62); min-height:95px;
}
.metric-label { color:#94a3b8; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { color:#f8fafc; font-family:'Manrope'; font-size:1.35rem; font-weight:800; margin-top:.3rem; }
.result {
    border-radius:18px; padding:1.25rem 1.4rem; margin:.5rem 0 1rem;
    border:1px solid rgba(148,163,184,.25); background:rgba(30,41,59,.68);
}
.result.positive { border-color:rgba(52,211,153,.45); background:rgba(6,78,59,.25); }
.result.negative { border-color:rgba(251,113,133,.45); background:rgba(136,19,55,.22); }
.result-title { font-family:'Manrope'; font-size:1.8rem; font-weight:800; margin:.2rem 0; }
.small-muted { color:#9caec9; font-size:.88rem; }
div.stButton > button {
    border:0; border-radius:12px; padding:.72rem 1.2rem; font-weight:700;
    background:linear-gradient(100deg,#5277f7,#6957d9); color:white;
    min-height:3rem; transition:transform .15s ease, filter .15s ease;
}
div.stButton > button:hover { filter:brightness(1.12); transform:translateY(-1px); color:white; }
div[data-testid="stTextArea"] textarea {
    background:#101a30; color:#f8fafc; border:1px solid #34445f; border-radius:14px;
}
hr { border-color:rgba(148,163,184,.16); }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model(model_id: str):
    """Download and cache the user's original Hugging Face model."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    return tokenizer, model

def readable_label(raw_label: str) -> str:
    label = str(raw_label).strip()
    normalized = re.sub(r"[\s_-]+", "", label).lower()
    if normalized in {"positive", "pos", "1", "label1"}:
        return "Positive"
    if normalized in {"negative", "neg", "0", "label0"}:
        return "Negative"
    if "positive" in normalized:
        return "Positive"
    if "negative" in normalized:
        return "Negative"
    return label.replace("_", " ").title()

def predict(text: str, tokenizer, model, generic_mapping: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    class_id = int(torch.argmax(probs).item())
    confidence = float(probs[class_id].item())
    config_labels = getattr(model.config, "id2label", {}) or {}
    raw_label = config_labels.get(class_id, config_labels.get(str(class_id), f"LABEL_{class_id}"))
    label = readable_label(raw_label)

    # If the model config only contains generic LABEL_0 / LABEL_1 names,
    # use the selectable binary mapping in the sidebar.
    if label.lower().replace(" ", "") in {f"label{class_id}", f"class{class_id}"}:
        if len(probs) == 2:
            if generic_mapping == "Class 0 = Negative, Class 1 = Positive":
                label = "Negative" if class_id == 0 else "Positive"
            else:
                label = "Positive" if class_id == 0 else "Negative"

    all_scores = []
    for idx, score in enumerate(probs.tolist()):
        raw = config_labels.get(idx, config_labels.get(str(idx), f"LABEL_{idx}"))
        nice = readable_label(raw)
        if nice.lower().replace(" ", "") in {f"label{idx}", f"class{idx}"}:
            if len(probs) == 2:
                if generic_mapping == "Class 0 = Negative, Class 1 = Positive":
                    nice = "Negative" if idx == 0 else "Positive"
                else:
                    nice = "Positive" if idx == 0 else "Negative"
            else:
                nice = f"Class {idx}"
        all_scores.append((nice, float(score)))
    return label, confidence, all_scores, class_id

st.markdown("""
<div class="hero">
  <div class="eyebrow">Your model · Your predictions</div>
  <h1>BERT Sentiment Studio</h1>
  <p>Understand the sentiment behind any sentence or review. This app loads your
  published BERT classifier directly from Hugging Face and runs predictions on the
  Streamlit server — no paid Hugging Face Inference Endpoint required.</p>
  <span class="pill">🤗 Hugging Face model</span>
  <span class="pill">⚡ Live prediction</span>
  <span class="pill">🔒 No API token required for a public model</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Model settings")
    st.caption("Model repository")
    st.code(MODEL_ID, language=None)
    generic_mapping = st.selectbox(
        "If the model uses generic LABEL_0 / LABEL_1 names",
        [
            "Class 0 = Negative, Class 1 = Positive",
            "Class 0 = Positive, Class 1 = Negative",
        ],
        help="This setting is only used when the model config does not contain readable class names.",
    )
    st.markdown("---")
    st.caption("The model is downloaded once when the app starts, then kept in memory while the app is active.")

left, right = st.columns([1.35, .85], gap="large")

with left:
    st.markdown("### Analyze a text")
    st.markdown('<div class="small-muted">Paste a movie review, product review, comment, or sentence below.</div>', unsafe_allow_html=True)
    def set_positive_example():
        st.session_state["sentiment_text"] = "The movie was excellent. The acting was brilliant and I enjoyed every minute!"

    def set_negative_example():
        st.session_state["sentiment_text"] = "The movie was boring, poorly written, and a complete waste of time."

    def clear_example():
        st.session_state["sentiment_text"] = ""

    if "sentiment_text" not in st.session_state:
        st.session_state["sentiment_text"] = ""

    user_text = st.text_area(
        "Text to analyze",
        placeholder="Example: The movie was excellent. The acting was brilliant and I enjoyed every minute!",
        height=190,
        label_visibility="collapsed",
        max_chars=5000,
        key="sentiment_text",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.button("Use positive example", use_container_width=True, on_click=set_positive_example)
    with c2:
        st.button("Use negative example", use_container_width=True, on_click=set_negative_example)
    with c3:
        st.button("Clear text", use_container_width=True, on_click=clear_example)

    analyze = st.button("✨  Analyze sentiment", type="primary", use_container_width=True)

with right:
    st.markdown("### How it works")
    st.markdown("""
    <div class="panel">
      <p><b>01 · Input</b><br><span class="small-muted">Enter the text you want to classify.</span></p>
      <hr>
      <p><b>02 · Tokenization</b><br><span class="small-muted">Your model's tokenizer converts text into BERT input tokens.</span></p>
      <hr>
      <p><b>03 · Prediction</b><br><span class="small-muted">Your fine-tuned BERT model calculates class probabilities.</span></p>
      <hr>
      <p><b>04 · Result</b><br><span class="small-muted">See the predicted sentiment and confidence scores.</span></p>
    </div>
    """, unsafe_allow_html=True)

if analyze:
    if not user_text.strip():
        st.warning("Please enter some text before running the analysis.")
    else:
        try:
            with st.spinner("Loading your Hugging Face model and analyzing the text..."):
                tokenizer, model = load_model(MODEL_ID)
                label, confidence, scores, class_id = predict(
                    user_text.strip(), tokenizer, model, generic_mapping
                )

            style = "positive" if label.lower() == "positive" else "negative" if label.lower() == "negative" else ""
            emoji = "😊" if style == "positive" else "🙁" if style == "negative" else "🧠"
            st.markdown("---")
            st.markdown("### Prediction")
            st.markdown(
                f'<div class="result {style}"><div class="small-muted">MODEL PREDICTION</div>'
                f'<div class="result-title">{emoji} {html.escape(label)}</div>'
                f'<div class="small-muted">Confidence: {confidence*100:.2f}% · Predicted class index: {class_id}</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("#### Class probabilities")
            for score_label, score in scores:
                st.write(f"**{score_label}** — {score*100:.2f}%")
                st.progress(score)
            st.caption("Confidence is the model's softmax probability, not a guarantee that the prediction is correct.")
        except Exception as exc:
            st.error("The model could not be loaded or the prediction failed.")
            st.code(f"{type(exc).__name__}: {exc}")
            st.info("Check that the Hugging Face repository is public and contains model weights, config.json, and tokenizer files.")

st.markdown("---")
st.markdown(
    f'<div class="small-muted">Model source: <a href="https://huggingface.co/{html.escape(MODEL_ID)}" target="_blank">Hugging Face · {html.escape(MODEL_ID)}</a>'
    ' &nbsp; · &nbsp; Built with Streamlit and Transformers</div>',
    unsafe_allow_html=True,
)
