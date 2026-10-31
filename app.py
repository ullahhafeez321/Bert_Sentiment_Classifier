
import streamlit as st
import torch
import json
import re
import html

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)

# =====================================
# CONFIGURATION
# =====================================

st.set_page_config(
    page_title="NLP Analysis Studio",
    page_icon="🧠",
    layout="wide"
)

SENTIMENT_MODEL = "ullahhafeez321/bert-sentiment-classifier"
NER_MODEL = "ullahhafeez321/Qwen-NER-Decoder"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
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
}

.hero {
    background: linear-gradient(120deg,#172554,#172033,#111827);
    padding: 35px;
    border-radius: 20px;
    border: 1px solid #334155;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-family: 'Space Grotesk';
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
    background: linear-gradient(90deg,#2563eb,#7c3aed);
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
""", unsafe_allow_html=True)


# =====================================
# LOAD BERT ENCODER
# =====================================

@st.cache_resource
def load_bert():

    tokenizer = AutoTokenizer.from_pretrained(
        SENTIMENT_MODEL,
        token=st.secrets.get("HF_TOKEN", None)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL,
        token=st.secrets.get("HF_TOKEN", None)
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# =====================================
# LOAD QWEN DECODER
# =====================================

@st.cache_resource
def load_qwen():

    tokenizer = AutoTokenizer.from_pretrained(
        NER_MODEL,
        token=st.secrets.get("HF_TOKEN", None)
    )

    model = AutoModelForCausalLM.from_pretrained(
        NER_MODEL,
        token=st.secrets.get("HF_TOKEN", None),
        torch_dtype=(
            torch.float16 if DEVICE == "cuda"
            else torch.float32
        )
    )

    model.to(DEVICE)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


# =====================================
# BERT SENTIMENT PREDICTION
# =====================================

def predict_sentiment(text):

    tokenizer, model = load_bert()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    prediction = torch.argmax(probabilities).item()

    id2label = model.config.id2label

    raw_label = id2label.get(
        prediction,
        id2label.get(str(prediction), f"LABEL_{prediction}")
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
            id2label.get(str(i), f"LABEL_{i}")
        )

        scores.append({
            "label": str(label),
            "score": float(score)
        })

    return {
        "sentiment": sentiment,
        "confidence": float(probabilities[prediction]),
        "scores": scores
    }


# =====================================
# QWEN NER PROMPT
# =====================================

def create_ner_prompt(text, tokenizer):

    instruction = f"""
You are a Named Entity Recognition system.

Extract named entities from the following sentence.

Entity types:
PER = Person
ORG = Organization
LOC = Location
MISC = Miscellaneous

Return ONLY a valid JSON array.

Required format:

[
  {{"text": "Albert Einstein", "type": "PER"}},
  {{"text": "Princeton University", "type": "ORG"}}
]

If no entities are found, return [].

Sentence:
{text}

Entities:
"""

    if getattr(tokenizer, "chat_template", None):

        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            pass

    return instruction


# =====================================
# PARSE NER OUTPUT
# =====================================

def parse_entities(output):

    start = output.find("[")

    if start == -1:
        return []

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(output)):

        char = output[i]

        if in_string:

            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False

        else:

            if char == '"':
                in_string = True

            elif char == "[":
                depth += 1

            elif char == "]":

                depth -= 1

                if depth == 0:

                    try:
                        data = json.loads(output[start:i+1])

                        if isinstance(data, list):
                            return [
                                {
                                    "text": str(item["text"]).strip(),
                                    "type": str(item.get("type", "ENTITY")).upper()
                                }
                                for item in data
                                if isinstance(item, dict) and item.get("text")
                            ]

                    except Exception:
                        return []

    return []


# =====================================
# QWEN ENTITY EXTRACTION
# =====================================

def extract_entities(text):

    tokenizer, model = load_qwen()

    prompt = create_ner_prompt(text, tokenizer)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(DEVICE)

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][input_length:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    entities = parse_entities(generated_text)

    return {
        "entities": entities,
        "raw_output": generated_text
    }


# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("🧠 NLP Studio")

    st.caption("Dual Transformer Intelligence")

    st.divider()

    st.markdown("### MODEL 01")

    st.success("BERT Encoder")

    st.caption(SENTIMENT_MODEL)

    st.divider()

    st.markdown("### MODEL 02")

    st.info("Qwen Decoder")

    st.caption(NER_MODEL)

    st.divider()

    st.markdown("### Runtime")

    st.write(f"Device: {DEVICE.upper()}")

    st.caption("Models load on first use.")


# =====================================
# MAIN WEBSITE
# =====================================

st.markdown("""
<div class="hero">

<div style="color:#93c5fd;letter-spacing:3px;font-size:12px;">
DUAL MODEL LANGUAGE INTELLIGENCE
</div>

<h1>🧠 NLP Analysis Studio</h1>

<p>
Analyze sentiment and extract named entities
using two independently trained Hugging Face models.
</p>

</div>
""", unsafe_allow_html=True)


# =====================================
# INPUT
# =====================================

st.subheader("Enter Your Text")

text = st.text_area(
    "Your sentence",
    placeholder="Example: Albert Einstein worked at Princeton University.",
    height=140,
    label_visibility="collapsed"
)

analyze = st.button(
    "✨ Analyze With Both Models",
    use_container_width=True
)


# =====================================
# RUN BOTH MODELS
# =====================================

if analyze:

    if not text.strip():

        st.warning("Please enter some text.")

    else:

        sentiment = None
        ner = None

        sentiment_error = None
        ner_error = None

        col1, col2 = st.columns(2)

        with st.spinner("Running BERT and Qwen..."):

            try:
                sentiment = predict_sentiment(text.strip())

            except Exception as e:
                sentiment_error = str(e)

            try:
                ner = extract_entities(text.strip())

            except Exception as e:
                ner_error = str(e)

        # =====================================
        # SENTIMENT RESULTS
        # =====================================

        with col1:

            st.subheader("💬 Sentiment Analysis")

            st.caption("BERT · Encoder-only")

            if sentiment_error:

                st.error("Sentiment model failed.")
                st.exception(Exception(sentiment_error))

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

                    <p>Confidence:
                    {sentiment["confidence"]:.2%}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("#### Class Probabilities")

                for item in sorted(
                    sentiment["scores"],
                    key=lambda x: x["score"],
                    reverse=True
                ):

                    st.write(
                        f'{item["label"]}: {item["score"]:.2%}'
                    )

                    st.progress(item["score"])


        # =====================================
        # NER RESULTS
        # =====================================

        with col2:

            st.subheader("🏷️ Named Entity Recognition")

            st.caption("Qwen · Decoder-only")

            if ner_error:

                st.error("NER model failed.")
                st.exception(Exception(ner_error))

            else:

                entities = ner["entities"]

                st.markdown(
                    f"""
                    <div class="panel">
                    <div style="color:#94a3b8;">
                    ENTITIES EXTRACTED
                    </div>

                    <div class="metric">
                    {len(entities)}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
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
                            unsafe_allow_html=True
                        )

                    st.markdown("#### Structured Results")

                    st.dataframe(
                        entities,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info("No entities returned.")

                with st.expander("View Qwen raw output"):

                    st.code(ner["raw_output"] or "(empty output)")


        # =====================================
        # ORIGINAL TEXT
        # =====================================

        st.divider()

        st.subheader("Original Input")

        st.info(text.strip())

        st.caption(
            "Predictions depend on model training and may not always be correct."
        )


# =====================================
# FOOTER
# =====================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#64748b;">
    NLP Analysis Studio |
    Hugging Face Transformers |
    Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
