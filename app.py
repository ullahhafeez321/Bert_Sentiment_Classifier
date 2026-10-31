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
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');

    .stApp {
        background: #0b1020;
        color: #edf2ff;
        font-family: 'DM Sans', sans-serif;
    }
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        background: linear-gradient(120deg, #172554, #172033, #111827);
        padding: 35px;
        border-radius: 20px;
        border: 1px solid #334155;
        margin-bottom: 25px;
    }
    .hero h1 {
        color: white;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 42px;
    }
    .hero p {
        color: #cbd5e1;
        font-size: 17px;
    }
    .panel {
        background: #121a2c;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }
    .metric {
        font-size: 32px;
        font-weight: 700;
        color: white;
    }
    .entity {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 9px 13px;
        margin: 5px;
        color: white;
    }
    .entity-type {
        color: #93c5fd;
        font-weight: bold;
        margin-right: 8px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        min-height: 48px;
        font-weight: bold;
    }
    section[data-testid="stSidebar"] {
        background: #0e1729;
    }
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
    st.caption("Dual Model Language Intelligence")
    st.divider()

    st.markdown("### MODEL 01")
    st.success("BERT Encoder")
    st.caption(SENTIMENT_MODEL)

    st.divider()

    st.markdown("### MODEL 02")
    st.info("Qwen Decoder · LoRA")
    st.caption(NER_ADAPTER)

    st.divider()
    st.markdown("### Runtime")
    st.write(f"Device: {DEVICE.upper()}")
    st.caption("Models load when first used.")

    if HF_TOKEN:
        st.caption("Hugging Face token detected.")
    else:
        st.caption("Public model access is being used.")


# ============================================================
# MAIN WEBSITE
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div style="color:#93c5fd;letter-spacing:3px;font-size:12px;">
            DUAL MODEL LANGUAGE INTELLIGENCE
        </div>
        <h1>🧠 NLP Analysis Studio</h1>
        <p>
            Analyze sentiment and extract named entities using two
            independently trained Hugging Face models.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Enter Your Text")

text = st.text_area(
    "Your sentence",
    placeholder="Example: Albert Einstein worked at Princeton University.",
    height=140,
    label_visibility="collapsed",
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
    """
    <div style="text-align:center;color:#64748b;">
        NLP Analysis Studio | Hugging Face Transformers | Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
