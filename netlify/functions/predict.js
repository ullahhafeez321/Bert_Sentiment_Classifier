// Netlify serverless proxy. Keep your Hugging Face token on the server;
// never put it in app.js or any other browser-delivered file.
exports.handler = async function (event) {
  const headers = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS'
  };
  if (event.httpMethod === 'OPTIONS') return { statusCode: 204, headers, body: '' };
  if (event.httpMethod !== 'POST') return { statusCode: 405, headers, body: JSON.stringify({ error: 'Method not allowed.' }) };

  const endpoint = process.env.HF_ENDPOINT_URL;
  const token = process.env.HF_TOKEN;
  if (!endpoint || !token) {
    return { statusCode: 500, headers, body: JSON.stringify({ error: 'Server is not configured. Add HF_ENDPOINT_URL and HF_TOKEN in Netlify environment variables.' }) };
  }

  try {
    const body = JSON.parse(event.body || '{}');
    const text = typeof body.text === 'string' ? body.text.trim() : '';
    if (!text) return { statusCode: 400, headers, body: JSON.stringify({ error: 'Please provide text to analyze.' }) };
    if (text.length > 5000) return { statusCode: 413, headers, body: JSON.stringify({ error: 'Text must be 5,000 characters or fewer.' }) };

    const upstream = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ inputs: text, parameters: { top_k: null }, options: { wait_for_model: true } })
    });
    const responseText = await upstream.text();
    let data;
    try { data = JSON.parse(responseText); } catch { data = { error: responseText.slice(0, 500) }; }
    if (!upstream.ok) {
      const detail = data && (data.error || data.message) ? (data.error || data.message) : `Hugging Face returned ${upstream.status}.`;
      return { statusCode: upstream.status >= 500 ? 502 : upstream.status, headers, body: JSON.stringify({ error: detail }) };
    }
    return { statusCode: 200, headers, body: JSON.stringify(data) };
  } catch (error) {
    return { statusCode: 500, headers, body: JSON.stringify({ error: 'Could not reach the Hugging Face model. ' + error.message }) };
  }
};
