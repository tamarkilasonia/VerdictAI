# Evaluation and tests

[← Back to the README](../README.md)

## How reliable is the audit?

On 13 hand-written cases (52 verdicts: correct facts, a wrong year, a wrong author, an
unverified side claim, a famous fact wrong without evidence, bias, an insult, an
off-topic answer, small talk, a follow-up that needs the conversation), the combined
audit by `openai/gpt-oss-120b` on Groq matched the expected verdict **50 / 52** times
(the four separate checks: 49 / 52). Every false fact was caught. The two misses were
debatable: "women are naturally worse at math" was also flagged as a hallucination and
as toxic, not only as bias.

- These are well-known facts, so the large auditor is right even without evidence.
  The evidence matters more for less-known facts, and it gives you the sources to
  check yourself. (The earlier local auditor, `llama3.2`, got 10/14 verdicts right
  without evidence and 12/14 with it)
- **End to end (real services):** "When did the Berlin Wall fall?": both passed, one
  answer shown. "Who won the 2026 FIFA World Cup?": both answers contradicted by Wikipedia, so the
  chat said "I don't know"; after **Revise answer**, Gemini's evidence-based answer
  passed and was shown
- **Agreement threshold:** on 16 simple factual questions, 14 answer pairs that meant the same scored 0.80–1.00; the two that scored 0.78 said
  the same but one was much longer. A real disagreement (the second model invented the
  author of *The Knight in the Panther's Skin*) scored 0.66. So 0.80 errs on the side
  of showing both answers

Two rules are enforced **in code** because small models ignore them in prompts:
a hallucination score is capped at 0.2 unless a claim is really *CONTRADICTED*
("the evidence does not mention X" is relabelled *UNVERIFIED*), and raised to at least
0.7 when one is.

## Fact check of attached PDFs

Three test PDFs with known mistakes, the list of their mistakes and the results are in
[`eval/`](../eval/README.md): 15 of 16 wrong claims were caught, with 3 false alarms.

## Tests

```bash
pytest -q
```

167 small tests cover the pure functions: cosine similarity and the agreement threshold,
parsing of the auditor's JSON (valid, fenced, invalid, out of range), verdict
thresholds at their exact boundaries, the hallucination label rules, passage
splitting, re-ranking and dropping off-topic evidence, the audit log (including "the
API key is never written"), the graph's routing decisions, the search keywords,
English-only prompts, auditing both answers (once each, even after a revision), choosing
the one answer to show (both pass, one fails, both fail and "I don't know", unchecked
answers never shown), Wikipedia's own answer (with the reason shown), the combined audit
and its fallback to separate checks, the low reasoning effort for gpt-oss, empty answers asked again, formulas rewritten for display, attached PDFs (reading, page and size limits, scanned, broken and password-protected files, page-numbered chunks, search, "not in the document" without asking the models, no Wikipedia in document mode, the log fields), the PDF fact check (claim extraction and cut-off replies, claims from every page, batches of 5, matching auditor issues to claims, confirmed / contradicted / can't be checked, waiting for the per-minute limit, warnings, the summary for "is this correct?"), no revision or retry
when the auditor's daily limit is used up, the two-service settings and the error messages (rate limit,
wrong key, unknown model, busy service). They use fakes instead of real AI services
and embeddings, so they need neither API keys nor internet, nor PyTorch. GitHub Actions
runs them on every push and pull request (`.github/workflows/tests.yml`).
