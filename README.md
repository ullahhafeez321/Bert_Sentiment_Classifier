# BERT Sentiment Studio — Netlify

A responsive sentiment-analysis website connected to your Hugging Face model:
`ullahhafeez321/bert-sentiment-classifier`.

## Important: enable model inference first

The Hugging Face model page currently says that the model is **not deployed by an Inference Provider**. A public model repository does not automatically provide a live prediction API. This project uses a Hugging Face **Inference Endpoint** so predictions run on your actual model. Create/deploy an endpoint for the model in Hugging Face, wait until it is running, and copy its endpoint URL.

Hugging Face guide: https://huggingface.co/docs/inference-endpoints/index
Your model: https://huggingface.co/ullahhafeez321/bert-sentiment-classifier

## Deploy to your existing Netlify site

1. Extract this ZIP.
2. Copy the contents of the `bert-sentiment-netlify` folder into the root of your existing Netlify site's GitHub repository. Keep your existing repository files backed up first. If it already has a `netlify.toml`, merge the `[build]` and `[functions]` settings rather than blindly replacing it.
3. Commit and push the changes. Netlify should build and deploy the site.
4. In Netlify, open **Site configuration → Environment variables** and add:
   - `HF_ENDPOINT_URL` = your running Hugging Face Inference Endpoint URL
   - `HF_TOKEN` = a Hugging Face access token with permission to call that endpoint
5. Trigger a new deploy so the function receives the environment variables.
6. Open your Netlify website and test a positive and a negative sentence.

## Security

The Hugging Face token is used only inside the Netlify serverless function. Do not add it to `app.js`, `index.html`, or any public frontend file. Do not commit secrets to GitHub.

## Local preview

Install Node.js, then from this folder run:

```bash
npm install -g netlify-cli
netlify dev
```

Set `HF_ENDPOINT_URL` and `HF_TOKEN` in a local `.env` file before running. Do not commit `.env`.

## Notes

- The site does not retrain or modify your model.
- It requests all class scores where supported by the endpoint and displays the returned labels as configured by the model.
- If the endpoint returns an error, the website displays the error instead of inventing a prediction.
