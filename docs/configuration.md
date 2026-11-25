# Setup and configuration

[← Back to the README](../README.md)

Requirements: Python 3.12 (3.11+ works), an internet connection and one free Groq API key.

### One AI service, three models

| Role | Model (on Groq) | What it does |
|---|---|---|
| Primary | `openai/gpt-oss-20b` | Main answer, revisions, Wikipedia search keywords, claims from an attached PDF |
| Secondary | `qwen/qwen3.8-27b` | A second, independent answer to compare with (a different model family) |
| Auditor | `openai/gpt-oss-120b` | The four audit checks and the PDF fact check |

Free key: [console.groq.com](https://console.groq.com) → *API Keys* (no credit card). All
three models are reached through Groq's **OpenAI-compatible API**, so one client library
(`langchain-openai`) talks to them, and the primary model uses the same key unless
`PRIMARY_API_KEY` is set.

The models were chosen by testing them (November 2026). The primary model was first
Google Gemini (`gemini-3.1-flash-lite`, 1–3 s per answer; `gemini-3.5-flash` was as good
but took 20–44 s), but Gemini's free tier was often overloaded (it answered every request
with `503` for hours), which made one question take minutes. On Groq, `openai/gpt-oss-20b`
answered five test questions correctly in 0.4 s on average, and a whole question takes
3.5–5 s. `qwen/qwen3.8-27b` is the second opinion (Groq retired
`llama-3.3-70b-versatile`); `openai/gpt-oss-120b` is the auditor. Gemini still works for
the primary model: put its three `PRIMARY_` lines in `.env` (they are in `.env.example`,
commented out). Model names change over time: if one stops working, pick a current one in
the Groq console and put it in `.env`.

The project first ran on two small local models (Ollama: `llama3.2`, `qwen2.5:3b`).
In a test of five English questions, local `qwen2.5:7b` and `llama3.1:8b` took 5–6 s
per answer (Groq: under 1 s) and got more facts wrong, so the models moved to cloud
services. The model factory still works with any
OpenAI-compatible server, including a local Ollama (`http://localhost:11434/v1`).

```bash
# 1. Create a virtual environment and install the dependencies
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # includes PyTorch, so it takes a few minutes

# 2. Create your settings file and paste your Groq key into it
cp .env.example .env                # SECONDARY_API_KEY

# 3. Run
streamlit run app.py                # opens http://localhost:8501
```

The first question downloads the embedding model (~470 MB) from Hugging Face; later
runs use the local copy.

### Settings (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `SECONDARY_API_KEY` | *(required)* | Groq key. Never logged or printed |
| `PRIMARY_API_KEY` | the Groq key | Key for the primary model; set it only for another service (e.g. Gemini) |
| `PRIMARY_MODEL` | `openai/gpt-oss-20b` | Main answer, revisions, the Wikipedia search keywords, PDF claims |
| `PRIMARY_BASE_URL` | Groq's OpenAI-compatible URL | Where the primary model is reached |
| `SECONDARY_MODEL` | `qwen/qwen3.8-27b` | Second opinion for the agreement check |
| `AUDITOR_MODEL` | `openai/gpt-oss-120b` | Runs the four audit checks (on Groq) |
| `SECONDARY_BASE_URL` | Groq's OpenAI-compatible URL | Where AI 2 is reached |
| `AGREEMENT_THRESHOLD` | `0.80` | Cosine similarity at or above which the models "agree" |
| `HISTORY_MESSAGES` | `6` | How many previous chat messages the models see |
| `WIKI_USER_AGENT` | `VerdictAi/0.1 (AI course project)` | Identifies the app to Wikipedia, as Wikimedia requires. Put your own contact here |
