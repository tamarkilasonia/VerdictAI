# Evaluation documents for the PDF fact check

Three short English PDFs with known mistakes, used to test the fact check that runs when a
file is attached (see step 9 in [How VerdictAi works](../docs/how-it-works.md)). Each file mixes correct
claims with wrong ones. Some correct claims are "tricky": they sound wrong or depend on a
detail, so they test for false alarms.

The lists below are only for grading the results. They are not used anywhere in the code or
in the prompts.

## How to reproduce

1. Run the app (`streamlit run app.py`) or open the [online version](https://verdic-t-ai.streamlit.app)
2. Attach one of the PDFs with the attach button in the input field
3. Wait for the fact check to finish and compare the table under the file name with the
   lists below. Asking *"Is this information correct?"* shows the same result as a summary

One check uses about 12,000 tokens of the auditor's free daily limit on Groq for 20 claims
(~6%), so run the files one after another, not repeatedly.

Results can differ slightly between runs: the claims are extracted by a model, and the
auditor judges each claim against the Wikipedia passages found for it.

## world_history_key_events.pdf

3 pages, 16 events, **6 wrong**.

| Claim in the document | Correct |
|---|---|
| Julius Caesar was assassinated in 34 BC | 44 BC |
| King Henry VIII agreed to the Magna Carta in 1215 | King John |
| Columbus reached the Americas in 1498 | 1492 (1498 was his third voyage) |
| The storming of the Bastille was on July 14, 1799 | 1789 |
| Buzz Aldrin was the first person to walk on the Moon | Neil Armstrong (Aldrin was second) |
| The Berlin Wall fell on November 9, 1991 | 1989 |

**Result (2026-11-25):** 20 claims checked, **5 of the 6 wrong claims caught** (Caesar,
Magna Carta, Bastille, Aldrin, Berlin Wall), each with the correct fact and its Wikipedia
source. Columbus "1498" was marked confirmed: Wikipedia's introduction only says that his
voyages took place in 1492–1504. **1 false alarm:** "the tallest human-made structure for
more than 3,800 years" (the Great Pyramid) was marked contradicted, because Wikipedia now
says "more than 3,700 years". The check took 79 s, 60 s of it waiting for Groq's
per-minute limit.

## science_exploration_facts.pdf

1 page, 20 claims, **7 wrong**.

| Claim in the document | Correct |
|---|---|
| Newton published the *Principia* in 1667 | 1687 |
| The Moon orbits at about 384,400 miles | 384,400 **km** |
| Mendel experimented on heredity with fruit flies | pea plants |
| DNA's double helix was described by Watson and Pauling | Watson and Crick |
| Amundsen reached the South Pole in December 1921 | 1911 |
| The Wright brothers' first flight lasted about 12 minutes | about 12 **seconds** |
| The Sahara is the largest desert in the world, larger than the polar deserts | the largest **hot** desert (Antarctica is larger) |

Correct but tricky: light from the Sun takes about 8 minutes to reach Earth; the Mariana
Trench is nearly 11 km deep; the Great Wall of China is more than 20,000 km long with all
its branches.

**Result (2026-11-25):** **7 of 7 wrong claims caught.** **2 false alarms:**
"Mendeleev published his first periodic table in 1869" (the passage found came from
*Types of periodic tables* and gave 1871) and "the ISS has been continuously occupied since
November 2000" (the passage from *Space station* named October, when Expedition 1 was
launched). In both cases the auditor followed a side article instead of the main one.

## landmarks_quick_check.pdf

1 page, 8 claims, **3 wrong**.

| Claim in the document | Correct |
|---|---|
| The Statue of Liberty was a gift from Spain | France |
| The Great Fire of London took place in 1766 | 1666 |
| The Panama Canal connects the Atlantic with the Indian Ocean | the Atlantic with the **Pacific** |

**Result (2026-11-25):** **8 of 8 correct verdicts** (5 confirmed, 3 contradicted), both
locally and on the online version.

## Summary

| File | Claims | Wrong | Caught | False alarms |
|---|---|---|---|---|
| world_history_key_events.pdf | 20 checked | 6 | 5 | 1 |
| science_exploration_facts.pdf | 20 | 7 | 7 | 2 |
| landmarks_quick_check.pdf | 8 | 3 | 3 | 0 |
| **Total** | **48** | **16** | **15** | **3** |
