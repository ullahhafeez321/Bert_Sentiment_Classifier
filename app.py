import html
import json
import os
import re

import streamlit as st
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from peft import PeftConfig, PeftModel


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NLP Analysis Studio",
    page_icon="🧠",
    layout="wide",
)

SENTIMENT_MODEL = "ullahhafeez321/bert-sentiment-classifier"
NER_ADAPTER = "ullahhafeez321/Qwen-NER-Decoder"
BASE_MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"
SENTIMENT_MODEL_URL = f"https://huggingface.co/{SENTIMENT_MODEL}"
NER_MODEL_URL = f"https://huggingface.co/{NER_ADAPTER}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_hf_token():
    """Read an optional Hugging Face token from Streamlit secrets or env."""
    try:
        return st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN") or None
    except Exception:
        return os.getenv("HF_TOKEN") or None


HF_TOKEN = get_hf_token()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root { color-scheme: dark; }
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(37,99,235,.16), transparent 34%),
            radial-gradient(ellipse at 90% 8%, rgba(124,58,237,.13), transparent 30%),
            #080d19;
        color: #e8eefb;
        font-family: 'DM Sans', sans-serif;
    }
    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3.5rem;
    }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.025em; }
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(115deg, rgba(23,37,84,.98), rgba(17,24,39,.96) 58%, rgba(49,30,88,.92));
        padding: clamp(25px, 4vw, 46px);
        border-radius: 24px;
        border: 1px solid rgba(148,163,184,.22);
        margin-bottom: 26px;
        box-shadow: 0 22px 70px rgba(0,0,0,.24);
    }
    .hero:after {
        content: '';
        position: absolute;
        width: 260px; height: 260px;
        right: -80px; top: -110px;
        border-radius: 50%;
        border: 1px solid rgba(191,219,254,.18);
        box-shadow: 0 0 0 28px rgba(191,219,254,.025), 0 0 0 58px rgba(191,219,254,.02);
    }
    .eyebrow { color: #93c5fd; letter-spacing: .22em; font-size: 11px; font-weight: 700; }
    .hero h1 { color: #fff; font-size: clamp(32px, 4vw, 48px); margin: 12px 0 8px 0; }
    .hero p { color: #cbd5e1; font-size: 16px; max-width: 760px; line-height: 1.7; margin-bottom: 0; }
    .model-card {
        background: linear-gradient(145deg, rgba(20,30,51,.96), rgba(13,21,37,.96));
        border: 1px solid rgba(100,116,139,.28);
        border-radius: 18px;
        padding: 21px;
        min-height: 178px;
        box-shadow: 0 12px 30px rgba(0,0,0,.12);
    }
    .model-kicker { color: #94a3b8; font-size: 11px; letter-spacing: .14em; font-weight: 700; }
    .model-card h3 { color: #f8fafc; font-size: 21px; margin: 9px 0 6px 0; }
    .model-card p { color: #aab8ce; font-size: 13px; line-height: 1.55; min-height: 40px; }
    .model-card a { color: #93c5fd !important; font-size: 13px; font-weight: 700; text-decoration: none; }
    .model-card a:hover { color: #c4b5fd !important; text-decoration: underline; }
    .panel {
        background: linear-gradient(145deg, rgba(18,27,46,.98), rgba(13,21,37,.98));
        padding: 24px;
        border-radius: 18px;
        border: 1px solid rgba(100,116,139,.3);
        margin-bottom: 15px;
        box-shadow: 0 12px 30px rgba(0,0,0,.12);
    }
    .metric { font-size: 34px; font-weight: 700; color: #fff; margin-top: 5px; }
    .entity {
        display: inline-block;
        background: rgba(30,41,59,.9);
        border: 1px solid #475569;
        border-radius: 10px;
        padding: 9px 13px;
        margin: 5px 5px 5px 0;
        color: #f8fafc;
    }
    .entity-type { color: #93c5fd; font-size: 11px; font-weight: 800; letter-spacing: .06em; margin-right: 8px; }
    .stTextArea textarea {
        background: #101a2d !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
        line-height: 1.65;
    }
    .stTextArea textarea:focus { border-color: #60a5fa !important; box-shadow: 0 0 0 1px #60a5fa !important; }
    .stButton > button {
        background: linear-gradient(100deg, #2563eb, #6d4aff);
        color: white;
        border: 1px solid rgba(191,219,254,.18);
        border-radius: 12px;
        min-height: 50px;
        font-weight: 700;
        transition: transform .18s ease, filter .18s ease;
    }
    .stButton > button:hover { filter: brightness(1.12); transform: translateY(-1px); color: white; }
    div[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #3b82f6, #8b5cf6); }
    section[data-testid="stSidebar"] { background: #0b1323; border-right: 1px solid rgba(100,116,139,.18); }
    section[data-testid="stSidebar"] a { color: #93c5fd !important; }
    div[data-testid="stAlert"] { border-radius: 12px; }
    .footer { text-align: center; color: #718096; font-size: 12px; padding: 8px 0 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD BERT ENCODER — SENTIMENT CLASSIFICATION
# ============================================================

@st.cache_resource(show_spinner="Loading BERT sentiment classifier...")
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained(
        SENTIMENT_MODEL,
        token=HF_TOKEN,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL,
        token=HF_TOKEN,
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ============================================================
# LOAD QWEN DECODER — LORA ADAPTER + BASE MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading Qwen base model and NER adapter...")
def load_qwen():
    # The Hugging Face repository is a PEFT/LoRA adapter, not a
    # standalone AutoModelForCausalLM checkpoint.
    adapter_config = PeftConfig.from_pretrained(
        NER_ADAPTER,
        token=HF_TOKEN,
    )
    base_model_name = adapter_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        token=HF_TOKEN,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        token=HF_TOKEN,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        NER_ADAPTER,
        token=HF_TOKEN,
    )

    model.eval()

    return tokenizer, model, base_model_name


# ============================================================
# BERT SENTIMENT PREDICTION
# ============================================================

def predict_sentiment(text):
    tokenizer, model = load_bert()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    prediction = int(torch.argmax(probabilities).item())
    id2label = model.config.id2label

    raw_label = id2label.get(
        prediction,
        id2label.get(str(prediction), f"LABEL_{prediction}"),
    )
    label_upper = str(raw_label).upper()

    if "POS" in label_upper:
        sentiment = "Positive"
    elif "NEG" in label_upper:
        sentiment = "Negative"
    elif "NEU" in label_upper:
        sentiment = "Neutral"
    else:
        sentiment = str(raw_label)

    scores = []
    for i, score in enumerate(probabilities):
        label = id2label.get(
            i,
            id2label.get(str(i), f"LABEL_{i}"),
        )
        scores.append({
            "label": str(label),
            "score": float(score.item()),
        })

    return {
        "sentiment": sentiment,
        "confidence": float(probabilities[prediction].item()),
        "scores": scores,
    }


# ============================================================
# QWEN PROMPT — MATCH THE FINE-TUNING FORMAT
# ============================================================

def create_ner_prompt(text, tokenizer):
    # This follows the combined NER + relation extraction instruction
    # used during the model's fine-tuning. The UI displays entities only.
    system_instruction = (
        "You are an information extraction system. "
        "Extract named entities and relations from English text. "
        "Return ONLY valid JSON in this exact structure: "
        '{"entities":[{"text":"...","type":"PERSON"}],'
        '"relations":[{"subject":"...","relation":"RELATION_TYPE",'
        '"object":"..."}]}. '
        "Use the entity labels PERSON, ORGANIZATION, LOCATION, DATE, "
        "TIME, MONEY, PERCENT, MISC, or UNKNOWN. "
        "If no entities or relations are found, use empty arrays. "
        "Do not include explanations or markdown."
    )

    user_message = (
        "Extract named entities and relations from this English sentence:\n"
        f"{text}"
    )

    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_message},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    # Fallback for tokenizers without a chat template.
    return (
        f"System: {system_instruction}\n"
        f"User: {user_message}\n"
        "Assistant:"
    )


# ============================================================
# PARSE QWEN JSON OUTPUT
# ============================================================

def extract_json_payload(output):
    """Extract the first complete JSON object or array from model output."""
    decoder = json.JSONDecoder()

    for index, char in enumerate(output):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
            if isinstance(value, (dict, list)):
                return value
        except json.JSONDecodeError:
            continue

    return None


def parse_entities(output):
    payload = extract_json_payload(output)

    if isinstance(payload, dict):
        raw_entities = payload.get("entities", [])
    elif isinstance(payload, list):
        # Supports older checkpoints/prompts that return a JSON array.
        raw_entities = payload
    else:
        raw_entities = []

    entities = []
    if not isinstance(raw_entities, list):
        return entities

    for item in raw_entities:
        if not isinstance(item, dict):
            continue

        entity_text = str(item.get("text", "")).strip()
        entity_type = str(item.get("type", "UNKNOWN")).strip().upper()

        if entity_text:
            entities.append({
                "text": entity_text,
                "type": entity_type or "UNKNOWN",
            })

    return entities


# ============================================================
# QWEN NAMED ENTITY EXTRACTION
# ============================================================

def extract_entities(text):
    tokenizer, model, base_model_name = load_qwen()
    prompt = create_ner_prompt(text, tokenizer)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    # With device_map="auto", use the input embedding's device rather
    # than assuming the entire model is on one device.
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return {
        "entities": parse_entities(generated_text),
        "raw_output": generated_text,
        "base_model": base_model_name,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🧠 NLP Studio")
    st.caption("Dual-model language intelligence")
    st.divider()

    st.markdown("#### MODEL 01 · SENTIMENT")
    st.success("BERT · Encoder-only")
    st.caption("English sentiment classification")
    st.markdown(f"[Open Hugging Face repository ↗]({SENTIMENT_MODEL_URL})")

    st.divider()
    st.markdown("#### MODEL 02 · ENTITY EXTRACTION")
    st.info("Qwen · Decoder-only + LoRA")
    st.caption("English named entity extraction")
    st.markdown(f"[Open adapter repository ↗]({NER_MODEL_URL})")
    st.markdown(f"[Open Qwen base model ↗]({BASE_MODEL_URL})")

    st.divider()
    st.markdown("#### RUNTIME")
    st.write(f"**Device:** `{DEVICE.upper()}`")
    st.caption("Models load on first use and are cached for the session.")
    if HF_TOKEN:
        st.caption("Hugging Face token detected.")
    else:
        st.caption("Using public Hugging Face model access.")

# ============================================================
# MAIN WEBSITE
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">DUAL MODEL LANGUAGE INTELLIGENCE</div>
        <h1>Understand text. Extract meaning.</h1>
        <p>
            A two-model NLP workspace that combines sentiment classification
            with named entity recognition. Each model analyzes your original
            text independently.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Your AI models")
model_col1, model_col2 = st.columns(2, gap="large")
with model_col1:
    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-kicker">MODEL 01 · ENCODER-ONLY</div>
            <h3>💬 BERT Sentiment Classifier</h3>
            <p>Predicts the sentiment of English text and reports the confidence
            scores for the available sentiment classes.</p>
            <a href="{SENTIMENT_MODEL_URL}" target="_blank">Explore model on Hugging Face ↗</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
with model_col2:
    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-kicker">MODEL 02 · DECODER-ONLY + LORA</div>
            <h3>🏷️ Qwen NER Decoder</h3>
            <p>Extracts named entities from English text using a fine-tuned
            Qwen base model and its LoRA adapter.</p>
            <a href="{NER_MODEL_URL}" target="_blank">Explore adapter on Hugging Face ↗</a>
            <span style="color:#64748b; padding:0 7px;">·</span>
            <a href="{BASE_MODEL_URL}" target="_blank">Base model ↗</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.markdown("### Try the analyzer")
st.caption("Enter an English sentence or a short paragraph. Both models receive the same original input.")

text = st.text_area(
    "Your text",
    placeholder="Example: Albert Einstein worked at Princeton University in New Jersey.",
    height=150,
    label_visibility="collapsed",
    key="analysis_input",
)

analyze = st.button(
    "✨ Analyze With Both Models",
    use_container_width=True,
)


# ============================================================
# RUN BOTH MODELS INDEPENDENTLY
# ============================================================

if analyze:
    clean_text = text.strip()

    if not clean_text:
        st.warning("Please enter some text.")
    else:
        sentiment = None
        ner = None
        sentiment_error = None
        ner_error = None

        col1, col2 = st.columns(2)

        with st.spinner("Running BERT sentiment classification and Qwen NER..."):
            # Each model receives the original user text independently.
            try:
                sentiment = predict_sentiment(clean_text)
            except Exception as exc:
                sentiment_error = str(exc)

            try:
                ner = extract_entities(clean_text)
            except Exception as exc:
                ner_error = str(exc)

        # ----------------------------------------------------
        # SENTIMENT RESULTS
        # ----------------------------------------------------

        with col1:
            st.subheader("💬 Sentiment Analysis")
            st.caption("BERT · Encoder-only")

            if sentiment_error:
                st.error("Sentiment model failed to run.")
                st.code(sentiment_error)
            else:
                st.markdown(
                    f"""
                    <div class="panel">
                        <div style="color:#94a3b8;">
                            PREDICTED SENTIMENT
                        </div>
                        <div class="metric">
                            {html.escape(sentiment["sentiment"])}
                        </div>
                        <p>Confidence: {sentiment["confidence"]:.2%}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### Class Probabilities")

                for item in sorted(
                    sentiment["scores"],
                    key=lambda item: item["score"],
                    reverse=True,
                ):
                    st.write(f'{item["label"]}: {item["score"]:.2%}')
                    st.progress(item["score"])

        # ----------------------------------------------------
        # NAMED ENTITY RESULTS
        # ----------------------------------------------------

        with col2:
            st.subheader("🏷️ Named Entity Recognition")
            st.caption("Qwen · Decoder-only · LoRA adapter")

            if ner_error:
                st.error("Qwen NER model failed to run.")
                st.code(ner_error)
            else:
                entities = ner["entities"]

                st.markdown(
                    f"""
                    <div class="panel">
                        <div style="color:#94a3b8;">
                            ENTITIES EXTRACTED
                        </div>
                        <div class="metric">{len(entities)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if entities:
                    for entity in entities:
                        st.markdown(
                            f"""
                            <span class="entity">
                                <span class="entity-type">
                                    {html.escape(entity["type"])}
                                </span>
                                {html.escape(entity["text"])}
                            </span>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("#### Structured Results")
                    st.dataframe(
                        entities,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info(
                        "No entities were returned. Check the raw output "
                        "below to see the model's response."
                    )

                with st.expander("View Qwen raw output"):
                    st.code(ner["raw_output"] or "(empty output)")
                    st.caption(f'Base model: {ner["base_model"]}')

        # ----------------------------------------------------
        # ORIGINAL TEXT
        # ----------------------------------------------------

        st.divider()
        st.subheader("Original Input")
        st.info(clean_text)
        st.caption(
            "Both models process the original input independently. "
            "Predictions depend on model training and may not always be correct."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown(
    f"""
    <div class="footer">
        NLP Analysis Studio &nbsp;·&nbsp; Built with Streamlit and Hugging Face<br>
        <a href="{SENTIMENT_MODEL_URL}" target="_blank">BERT model</a>
        &nbsp;·&nbsp;
        <a href="{NER_MODEL_URL}" target="_blank">Qwen NER adapter</a>
    </div>
    """,
    unsafe_allow_html=True,
)
