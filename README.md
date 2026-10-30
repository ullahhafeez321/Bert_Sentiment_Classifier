# BERT Sentiment Studio — Streamlit

A Streamlit app that loads your **existing public Hugging Face model** directly:

`ullahhafeez321/bert-sentiment-classifier`

It uses the original Transformers/PyTorch model files. It does **not** retrain, convert to ONNX, or require a paid Hugging Face Inference Endpoint.

## Deploy free with Streamlit Community Cloud

1. Download and extract this ZIP.
2. Create a new **public GitHub repository** (or use a separate folder/repository so you do not overwrite your existing Netlify website).
3. Upload these files to the repository root:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `runtime.txt`
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Click **Create app**.
6. Select your repository, branch (`main`), and main file path (`app.py`).
7. Click **Deploy**.

Streamlit will install the packages, download your model from Hugging Face, and start the app. The first model download can take a few minutes.

## Model details

The app loads the model with:

```python
AutoTokenizer.from_pretrained("ullahhafeez321/bert-sentiment-classifier")
AutoModelForSequenceClassification.from_pretrained("ullahhafeez321/bert-sentiment-classifier")
```

The model is cached in Streamlit memory while the app is running. A public model does not need an HF token.

## Important notes

- Streamlit Community Cloud has a free tier, but it has resource limits and apps can go to sleep after inactivity. The first visitor after sleep may need to wait for it to wake up.
- This is a separate Streamlit app URL (`*.streamlit.app`). Netlify static hosting cannot run a persistent Python/PyTorch Streamlit server. You can keep your existing Netlify site and add a link to the Streamlit app after deployment.
- The app runs on CPU. Prediction speed depends on the available free resources and model size.
- If your model's `config.json` uses generic class names such as `LABEL_0` and `LABEL_1`, choose the correct class mapping in the sidebar. The default assumes class 0 = Negative and class 1 = Positive. Verify this against the label mapping used during your model training.
- The app displays the class names from your model configuration when they are available.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Change the model repository

Set the environment variable `HF_MODEL_ID` to another public Hugging Face model ID if needed. By default, it uses your model.
