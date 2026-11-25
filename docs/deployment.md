# Deployment

[← Back to the README](../README.md)

**Online version: [https://verdic-t-ai.streamlit.app](https://verdic-t-ai.streamlit.app)**

- Hosted on **Streamlit Community Cloud**, deployed from this GitHub repository (`main`,
  `app.py`). Every push to `main` redeploys the app
- **Python 3.12**, with the packages in `requirements.txt`
- The API keys are not in the repository: they are stored in the app's **Secrets** on
  Streamlit Cloud, which passes them to the app as environment variables (read the same
  way as `.env`). The secrets used are `PRIMARY_API_KEY`, `PRIMARY_MODEL`,
  `PRIMARY_BASE_URL`, `SECONDARY_API_KEY`, `SECONDARY_MODEL`, `AUDITOR_MODEL`,
  `SECONDARY_BASE_URL` and `WIKI_USER_AGENT`
- The online version runs **entirely on Groq**: `PRIMARY_MODEL=openai/gpt-oss-20b`,
  `PRIMARY_BASE_URL=https://api.groq.com/openai/v1` (second opinion
  `qwen/qwen3.8-27b`, auditor `openai/gpt-oss-120b`)
- When nobody has used it for a while, the app goes to sleep; the first visit after that
  wakes it up, which takes a minute or so (it also loads the embedding model)
- The online version shares the free Groq limits described under
  [Known limitations](limitations-and-future-work.md#known-limitations)
