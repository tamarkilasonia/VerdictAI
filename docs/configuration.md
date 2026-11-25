# Setup and configuration

[← Back to the README](../README.md)

Requirements: Python 3.11+ (3.12 recommended), an internet connection and two free API keys.

### Two AI services, both free

| Role | Service | Free key | Why |
|---|---|---|---|
| Primary answer, revisions, search keywords | **Google Gemini** | [aistudio.google.com](https://aistudio.google.com) → *Get API key* | Fast, good answers |
| Second opinion + the four audit checks | **Groq** | [console.groq.com](https://console.groq.com) → *API Keys* | Higher free limits for the many audit calls; a different company gives an independent second opinion |

Both are reached through their **OpenAI-compatible API**, so one client library
(`langchain-openai`) talks to both. No credit card is needed for either free tier.

The default models were chosen by testing them (November 2026):
`gemini-3.1-flash-lite` answers in 1–3 s (`gemini-3.5-flash` was as good but took
20–44 s per answer); `qwen/qwen3.8-27b` is the second opinion (Groq
retired `llama-3.3-70b-versatile`); `openai/gpt-oss-120b` is the auditor. Model names
change over time: if one stops working, pick a current one in Google AI Studio or the
Groq console and put it in `.env`.

When Gemini's free tier is overloaded (it answered every request with `503` for hours in
November 2026), the app runs entirely on Groq by pointing the `PRIMARY_` settings at it:
`PRIMARY_MODEL=openai/gpt-oss-20b`, `PRIMARY_BASE_URL=https://api.groq.com/openai/v1` and
the Groq key. Measured: a question takes 3.5–5 s in total.

The project first ran on two small local models (Ollama: `llama3.2`, `qwen2.5:3b`).
In a test of five English questions, local `qwen2.5:7b` and `llama3.1:8b` took 5–6 s
per answer (Groq: under 1 s) and got more facts wrong, so the models moved to these
two services. The model factory still works with any
OpenAI-compatible server, including a local Ollama (`http://localhost:11434/v1`).

```bash
# 1. Create a virtual environment and install the dependencies
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # includes PyTorch, so it takes a few minutes

# 2. Create your settings file and paste your two keys into it
cp .env.example .env                # PRIMARY_API_KEY (Gemini), SECONDARY_API_KEY (Groq)

# 3. Run
streamlit run app.py                # opens http://localhost:8501
```

The first question downloads the embedding model (~470 MB) from Hugging Face; later
runs use the local copy.

### Settings (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `PRIMARY_API_KEY` | *(required)* | Gemini key. Never logged or printed |
| `PRIMARY_MODEL` | `gemini-3.1-flash-lite` | Main answer, revisions and the Wikipedia search keywords |
| `PRIMARY_BASE_URL` | Gemini's OpenAI-compatible URL | Where AI 1 is reached |
| `SECONDARY_API_KEY` | *(required)* | Groq key. Never logged or printed |
| `SECONDARY_MODEL` | `qwen/qwen3.8-27b` | Second opinion for the agreement check |
| `AUDITOR_MODEL` | `openai/gpt-oss-120b` | Runs the four audit checks (on Groq) |
| `SECONDARY_BASE_URL` | Groq's OpenAI-compatible URL | Where AI 2 is reached |
| `AGREEMENT_THRESHOLD` | `0.80` | Cosine similarity at or above which the models "agree" |
| `HISTORY_MESSAGES` | `6` | How many previous chat messages the models see |
| `WIKI_USER_AGENT` | `VerdictAi/0.1 (AI course project)` | Identifies the app to Wikipedia, as Wikimedia requires. Put your own contact here |
