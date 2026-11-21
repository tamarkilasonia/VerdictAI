"""Parsing the auditor's output, verdict thresholds and the score/label rules."""

import pytest

from auditor import (
    CheckOutcome,
    CheckResult,
    Issue,
    align_score_with_labels,
    build_check_input,
    describe_problems,
    needs_revision,
    parse_check_output,
    pick_answer,
    verdict,
)

# ------------------------------------------------------------------ parsing


def test_parses_clean_json():
    result = parse_check_output('{"score": 0.4, "issues": [{"quote": "x", "reason": "y"}]}')
    assert result.score == 0.4
    assert result.issues == [Issue(quote="x", reason="y")]


def test_parses_json_inside_code_fence_and_chatter():
    text = 'Sure! Here it is:\n```json\n{"score": 0.1, "issues": []}\n```\nHope this helps.'
    assert parse_check_output(text) == CheckResult(score=0.1, issues=[])


def test_missing_issues_defaults_to_empty_list():
    assert parse_check_output('{"score": 0}').issues == []


@pytest.mark.parametrize(
    "bad_output",
    [
        "not json at all",
        '{"score": "high"}',  # not a number
        '{"score": 85}',  # outside 0..1
        '{"score": -0.1}',
        '{"issues": []}',  # score missing
        '{"score": 0.5, "issues": [{"quote": "x"}]}',  # issue without reason
    ],
)
def test_invalid_output_returns_none(bad_output):
    assert parse_check_output(bad_output) is None


# ------------------------------------------------------------------ verdicts


@pytest.mark.parametrize(
    "score, expected",
    [(0.0, "pass"), (0.29, "pass"), (0.3, "warn"), (0.69, "warn"), (0.7, "fail"), (1.0, "fail")],
)
def test_verdict_when_lower_is_better(score, expected):
    for check in ("hallucination", "bias", "toxicity"):
        assert verdict(check, score) == expected


@pytest.mark.parametrize(
    "score, expected",
    [(1.0, "pass"), (0.71, "pass"), (0.7, "warn"), (0.31, "warn"), (0.3, "fail"), (0.0, "fail")],
)
def test_verdict_for_intent_alignment_higher_is_better(score, expected):
    assert verdict("intent_alignment", score) == expected


# ------------------------------------------------ hallucination label rules


def test_unverified_only_is_capped():
    result = CheckResult(score=0.9, issues=[Issue(quote="x", reason="UNVERIFIED: not in evidence")])
    assert align_score_with_labels(result).score == 0.2


def test_real_contradiction_raises_score():
    # The auditor's words say "contradicted" but its number says 0: trust the words.
    result = CheckResult(
        score=0.0, issues=[Issue(quote="by van Gogh", reason="CONTRADICTED: evidence says Leonardo")]
    )
    assert align_score_with_labels(result).score == 0.7


def test_mislabelled_not_mentioned_becomes_unverified():
    issue = Issue(quote="in 1928", reason="CONTRADICTED: the evidence does not mention the year.")
    fixed = align_score_with_labels(CheckResult(score=1.0, issues=[issue]))
    assert fixed.score == 0.2
    assert fixed.issues[0].reason.startswith("UNVERIFIED")


# ------------------------------------------------------- revision helpers


def outcome(check, score):
    return CheckOutcome(check, result=CheckResult(score=score))


def test_needs_revision_only_on_fail():
    assert needs_revision({"bias": outcome("bias", 0.9)})
    assert not needs_revision({"bias": outcome("bias", 0.5)})  # ⚠️ is not ❌


def test_unavailable_check_does_not_trigger_revision():
    assert not needs_revision({"bias": CheckOutcome("bias", error="audit unavailable")})


def test_describe_problems_skips_passing_checks():
    audit = {
        "bias": CheckOutcome("bias", CheckResult(score=0.9, issues=[Issue(quote="q", reason="r")])),
        "toxicity": outcome("toxicity", 0.0),
    }
    assert describe_problems(audit) == ['bias: "q": r']


def test_hallucination_input_puts_evidence_before_answer():
    text = build_check_input("hallucination", "Q?", "the answer", [], ["fact one"])
    assert text.index("EVIDENCE") < text.index("ANSWER")


def test_hallucination_input_without_evidence_has_no_evidence_section():
    assert "EVIDENCE" not in build_check_input("hallucination", "Q?", "A", [], [])


# ------------------------------------------------- choosing the answer to show


def _audit(**statuses):
    """An audit with the given check -> score."""
    return {check: CheckOutcome(check, CheckResult(score=score)) for check, score in statuses.items()}


def test_pick_answer_prefers_primary_when_both_pass():
    assert pick_answer([_audit(hallucination=0.0), _audit(hallucination=0.0)]) == 0


def test_pick_answer_skips_an_answer_with_any_failed_check():
    # Toxic (fail) PRIMARY, clean SECONDARY.
    assert pick_answer([_audit(hallucination=0.0, toxicity=0.9), _audit(hallucination=0.0)]) == 1
    # Off-topic PRIMARY: intent alignment 0.1 is a fail (high = good for intent).
    off_topic = _audit(hallucination=0.0, intent_alignment=0.1)
    assert pick_answer([off_topic, _audit(hallucination=0.0, intent_alignment=1.0)]) == 1


def test_pick_answer_prefers_fewer_warnings():
    # PRIMARY passes but its bias is moderate (warn); SECONDARY is clean.
    assert pick_answer([_audit(hallucination=0.0, bias=0.5), _audit(hallucination=0.0, bias=0.0)]) == 1


def test_pick_answer_returns_none_when_every_answer_failed():
    assert pick_answer([_audit(hallucination=1.0), _audit(hallucination=0.9)]) is None
    assert pick_answer([]) is None


def test_an_unchecked_answer_is_never_shown():
    # The hallucination check could not run (rate limit): the facts are unknown.
    unchecked = {"hallucination": CheckOutcome("hallucination", None, error="rate limit")}
    assert pick_answer([unchecked, _audit(hallucination=0.2)]) == 1
    assert pick_answer([unchecked]) is None
    assert pick_answer([_audit(bias=0.0)]) is None  # no hallucination check at all


def test_an_unavailable_minor_check_counts_as_a_warning():
    partly = _audit(hallucination=0.0)
    partly["bias"] = CheckOutcome("bias", None, error="rate limit")
    assert pick_answer([partly, _audit(hallucination=0.0, bias=0.0)]) == 1
    assert pick_answer([partly]) == 0
