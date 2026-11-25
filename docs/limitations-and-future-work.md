# Known limitations and future work

[← Back to the README](../README.md)

## Known limitations

- **Free tiers.** Gemini and Groq limit requests per minute and per day, and the limits
  change over time (Groq, November 2026: 1000 requests per day and 8000 tokens per
  minute for each model, and only 1000 *output* tokens per minute for Qwen). The tightest
  limit is the **auditor's 200,000 tokens per day**: one question's two audits use
  about 6,600, so roughly **30 questions a day** can be fully fact-checked. After that,
  the chat shows Wikipedia's text instead of unchecked AI answers, and the limit frees
  up gradually over the next hours. One question uses 5 requests (2 answers, 1 search query, 1 audit
  request for each answer). When a limit is hit or a service is down, the app retries once after about 2 s and
  gives up on a request after 15 s, then says so instead of crashing (it used to retry
  for ~30 s per request, which made one question take minutes while Gemini was
  overloaded). Free tiers may also use what you send to improve their services,
  so do not type personal information
- **Internet required.** Both AI services and Wikipedia are online
- **Agreement ≠ truth.** Embeddings capture topic and meaning, not exact facts:
  "water boils at 100 °C" and "…at 90 °C" score 0.94. Two models can also agree on the
  same mistake. That is why both answers are audited against the evidence
- **Wikipedia coverage.** The check uses English Wikipedia, and only page introductions, searched
  with a few keywords. Recent events, niche topics and non-encyclopedic questions
  (e.g. coding) often get weak or no evidence, and then the check falls back to
  general knowledge
- **Only as good as the evidence.** "Fact-checked" means no claim was contradicted
  by the Wikipedia passages found (or, without evidence, by the auditor's general
  knowledge). A claim the evidence does not mention is *unverified*, not proven; the
  issues list shows such claims. Both models can also be wrong in the same way, and
  the auditor itself can make mistakes
- **English only.** The models are told to answer in English, whatever language the
  question uses, and only English Wikipedia is searched. The embedding model cuts texts
  after ~128 tokens, so long answers are compared mostly by their beginning
- **Memory lives in RAM.** Conversations disappear when the server restarts
  (databases are out of scope for this project)
- **Speed.** Depends on the services' load; free tiers may also slow down or pause
  requests when the per-minute limit is reached

## Future work

- **Caching results.** Keep the answers, Wikipedia passages and audits of questions that
  were already asked (and fact checks of files already seen) in a cache, so a repeated
  question costs no AI request and stays within Groq's free limits
- **A per-user limit in the public app.** The online version shares one Groq key, so one
  visitor can use up the daily limit for everyone. A small limit per session (for example
  a number of questions and fact checks per hour) would keep the app usable for all
- **Georgian support again.** The app answered in Georgian earlier but was made
  English-only, because the free models wrote poor Georgian and Georgian Wikipedia is much
  smaller. Stronger multilingual models would make it worth bringing back
- **Sources beyond Wikipedia.** Wikipedia's introductions do not cover recent events,
  niche topics or exact details (e.g. Columbus's 1498 voyage). Other trusted sources, such
  as encyclopedias, news archives or a web search used only when Wikipedia is silent,
  would leave fewer claims "can't be checked"
- **Preferring the main article.** Two false alarms came from side articles (*Types of
  periodic tables* instead of *Periodic table*, *Space station* instead of *International
  Space Station*). Ranking the article whose title matches the claim's topic above related
  articles would avoid this
- **Memory in a database.** Conversations and fact checks live in RAM and disappear when
  the server restarts. LangGraph's SQLite or Postgres checkpointer would keep them, and
  would let several app instances share them
