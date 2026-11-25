# How VerdictAi works

[← Back to the README](../README.md)

## What happens when you ask a question

1. **Two answers in parallel.** Two AI services get the question plus the last few chat
   messages at the same time: a **primary** model (Google Gemini by default, or a Groq
   model, see [setup](configuration.md)) and **Groq** (secondary, Qwen)
2. **Agreement check.** Both answers are embedded with
   `paraphrase-multilingual-MiniLM-L12-v2` (local, free) and compared with cosine
   similarity (threshold 0.80). The score is shown in the analysis panel
3. **Evidence.** The primary model turns your question into a few keywords (Wikipedia's
   search needs every word to match, so a whole sentence often finds nothing), and
   English Wikipedia is searched with them. The page introductions are split into
   short passages and **re-ranked** by similarity to the answer; off-topic passages are
   dropped. The top 3 passages become the evidence (shown with links)
4. **Audit of both answers.** Four checks run on Groq for **each** answer, each
   returning a validated JSON result: `score` (0–1) plus `issues` (`quote` +
   `reason`). All four go to the auditor in **one** request per answer: four separate
   requests hit Groq's per-minute token limit and made the audit wait 18–37 s. If the
   combined reply cannot be parsed, the four checks run as separate requests instead
   (on 13 test cases the combined audit matched the expected verdicts 50/52 times,
   the separate checks 49/52). The panel compares the two in one table and shows both answers in full

   | Check | Looks at | Good | Moderate | Serious |
   |---|---|---|---|---|
   | Hallucination | answer vs. evidence | < 0.3 | 0.3–0.7 | ≥ 0.7 |
   | Bias | answer | < 0.3 | 0.3–0.7 | ≥ 0.7 |
   | Toxicity | answer | < 0.3 | 0.3–0.7 | ≥ 0.7 |
   | Intent alignment | recent conversation + answer | > 0.7 | 0.3–0.7 | ≤ 0.3 |

5. **One answer in the chat.** An AI answer may be shown only if its facts **were
   checked** and **none** of its checks is *Serious*; a contradicted fact always makes
   the hallucination check Serious. If both pass, the one with fewer warnings and fewer
   unverified details is shown (the primary on a tie). If one fails, only the other is shown.
   If **no AI answer passed** (or the auditor could not run), the chat shows
   **Wikipedia's own text**, quoted with a link: the first sentences of the page closest
   to the question, plus its most relevant passage. Only if Wikipedia has nothing
   relevant either does the chat say *"I don't know a reliable answer"*. Example: the
   auditor was over its limit and Gemini claimed Akaki Tsereteli wrote the poem
   *A Happy Nation*; the chat instead quoted Wikipedia, which names Ilia Chavchavadze. The
   models remember only what you saw, so a follow-up never builds on a wrong answer
6. **Human-in-the-loop revision.** When the AI answers were checked and none passed, the workflow **pauses** and
   asks you. Click **Revise answer** and the primary model rewrites its answer from the audit's
   issues and the evidence; the new answer is audited again (at most 2 revisions)
7. **Audit trail.** Every interaction is appended to `logs/audit_log.jsonl`
   (downloadable with the *Download log* button in the header)
8. **Answers from your own PDF.** Click the attach button in the input field and attach one English,
   text-based PDF (up to 50 pages, 10 MB). Its name appears above the field with a button to
   remove it. While it is attached, the text is split into short chunks that remember
   their page, and the 4 chunks closest to your question go to both models, which must
   answer only from them and cite the page, e.g. *(report.pdf, page 4)*. The auditor
   checks the answers against the same chunks instead of Wikipedia. If no chunk is
   close to the question, the models are not asked and the chat says the document does
   not cover it. Scanned PDFs, password-protected or broken files and files over the
   limits get a clear message. The file stays only in the browser session's memory.
   Tested on *Attention Is All You Need* (15 pages): "What BLEU score did the big
   Transformer get…?" → 28.4 *(page 8)*, both answers passed; "Who won the 2018 FIFA
   World Cup?" → not in the document, no model asked; "How much did it cost in dollars
   to train the model?" (close to the topic, not in the paper) → both models said the
   document does not contain it. Answers about the document begin with *"The document
   says…"*, so they are not read as a confirmation that the fact is true
9. **Fact check of the attached PDF.** Right after a file is attached, its factual claims
   are checked against Wikipedia, with a progress bar (*Checking facts… 7 / 16*). The
   primary model lists the claims (one per sentence, with its page); up to 20 are
   checked, taken from all pages. Each claim gets its own Wikipedia passages, and the
   existing auditor checks 5 claims per request to save Groq's limit. The result is a
   table under the file name: page, claim, confirmed / contradicted / can't be
   checked, the correct fact and the source. *Can't be checked* (Wikipedia neither
   confirms nor denies it) is kept apart from *contradicted*. Results are kept for the
   session, so attaching the same file again does not repeat the check. If an answer
   repeats a wrong claim, a warning with the correct fact and the source appears under
   it (found with the local embedding model, no new AI call), e.g. *The document says
   "…in 34 BC", but this is wrong: … 15 March 44 BC (source: Assassination of Julius
   Caesar)*. Asking *"Is this information correct?"* or *"Check this file"* shows the
   fact check's summary instead of searching the text. Tested on a 3-page history
   document with 16 events, 6 of them wrong: 5 of the 6 were caught (Caesar, Magna Carta,
   Bastille, Aldrin, Berlin Wall); Columbus "1498" passed, because Wikipedia's
   introduction only says his voyages took place in 1492–1504. One correct claim was
   flagged: the document says the Great Pyramid was the tallest structure "for more
   than 3,800 years", Wikipedia says "more than 3,700". The check of 20 claims used
   about 12,000 auditor tokens (~6% of the daily limit) and took 79 s, 60 s of it
   waiting for Groq's per-minute limit. There is no web search: what Wikipedia cannot
   settle stays *can't be checked*. Two more test files (2026-11-25): a science document
   with 20 claims, 7 of them wrong, had all 7 caught, with 2 false alarms (Mendeleev 1869
   and the ISS since November 2000, where the auditor followed a side article); a short
   landmarks document with 8 claims, 3 of them wrong, got all 8 verdicts right, locally
   and online. In total 15 of 16 wrong claims were caught, with 3 false alarms. The test
   files, the list of their mistakes and the results are in [`eval/`](../eval/README.md)

## Workflow details

The workflow diagram is in the [README](../README.md#workflow-langgraph).

- **State:** a typed `ChatState` with long-term fields (`messages`, `turns`, which append
  through reducers) and per-question fields (answers, evidence, audit, revisions,
  latency), which are reset at every new question
- **Memory:** a `MemorySaver` checkpointer with one `thread_id` per browser tab. Only the
  last `HISTORY_MESSAGES` messages are sent to the models
- **Human-in-the-loop:** `human_review` calls LangGraph's `interrupt()`. Because the
  paused run is stored in the checkpointer, it survives Streamlit's reruns; the
  button resumes it with `Command(resume="revise")`. Asking a new question instead
  resumes it with `"accept"`, which keeps the previous answer
