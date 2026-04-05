"""
eval/scorer.py

Computes F1 scores and detailed results for a single eval case.

Scoring model:
  - Ground truth = set of (cpt, modifier) tuples from the golden case
  - Prediction   = set of (cpt, modifier) tuples from the agent output
  - TP = codes correctly identified with correct modifier
  - FP = codes the agent produced that weren't expected (hallucinations)
  - FN = expected codes the agent missed

Modifier correctness:
  - (96127, None) and (96127, "59") are treated as distinct items
  - Missing a modifier counts as a false negative on the modifier pair
    AND a false positive on the bare code (since the bare code was predicted
    but the modifier pair was not)

Informational metadata stored alongside F1:
  - Per-code confidence scores
  - Overall traffic light status
  - Latency
  - Which codes were false positives / false negatives
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from golden_cases import GoldenCase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class CodeResult:
    """Scoring result for a single predicted code."""
    cpt: str
    modifier: Optional[str]
    outcome: str          # "true_positive" | "false_positive" | "false_negative"
    confidence: Optional[float] = None
    status: Optional[str] = None   # GREEN | YELLOW | RED
    citation: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.cpt}-{self.modifier}" if self.modifier else self.cpt

    def __repr__(self) -> str:
        return str(self)


@dataclass
class MustNotIncludeViolation:
    """A hallucinated code that should never appear."""
    cpt: str
    modifier: Optional[str]
    confidence: Optional[float] = None


@dataclass
class EvalResult:
    """Full scoring result for one case against one agent response."""
    case_id: str
    patient_id: str
    payer: str

    # F1 components
    true_positives: list[CodeResult] = field(default_factory=list)
    false_positives: list[CodeResult] = field(default_factory=list)
    false_negatives: list[CodeResult] = field(default_factory=list)

    # Hallucination violations
    must_not_include_violations: list[MustNotIncludeViolation] = field(default_factory=list)

    # Modifier-specific results
    missing_modifiers: list[tuple[str, str]] = field(default_factory=list)

    # Informational metadata
    overall_status: Optional[str] = None
    overall_confidence: Optional[float] = None
    latency_seconds: Optional[float] = None
    agent_claim_id: Optional[str] = None
    error: Optional[str] = None

    # Computed metrics
    @property
    def precision(self) -> float:
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        tp = len(self.true_positives)
        fn = len(self.false_negatives)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def passed(self) -> bool:
        """Case passes if F1 >= 0.80 and no must_not_include violations."""
        return self.f1 >= 0.80 and len(self.must_not_include_violations) == 0

    def summary_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        violations = f" [{len(self.must_not_include_violations)} hallucination(s)]" \
            if self.must_not_include_violations else ""
        return (
            f"{status} | {self.case_id} | "
            f"F1={self.f1:.3f} P={self.precision:.3f} R={self.recall:.3f}"
            f"{violations}"
        )


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def parse_agent_codes(
    agent_output: dict,
) -> list[tuple[str, Optional[str], float, str, Optional[str]]]:
    """
    Extract (cpt, modifier, confidence, status, citation) tuples
    from the agent output.

    Handles two formats:
      1. ClaimAuditLog JSON: codes[].cpt_code / included_in_claim / confidence.final_score
      2. Markdown-parsed:    codes[].cpt / status (no confidence)
    """
    results = []
    source = agent_output.get("source", "json")

    for code in agent_output.get("codes", []):
        # ── Format 1: ClaimAuditLog JSON ──────────────────────────────────
        if "cpt_code" in code:
            if not code.get("included_in_claim", True):
                continue
            cpt        = code.get("cpt_code", "")
            modifier   = code.get("modifier") or None
            confidence = code.get("confidence", {}).get("final_score", 0.0)
            status     = code.get("status", "UNKNOWN")
            citation   = code.get("citation")

        # ── Format 2: Markdown-parsed ──────────────────────────────────────
        else:
            cpt        = code.get("cpt", "")
            modifier   = code.get("modifier") or None
            confidence = None
            status     = code.get("status", "UNKNOWN")
            citation   = None

        if not cpt:
            continue

        results.append((cpt, modifier, confidence, status, citation))

    if source == "markdown":
        logger.info(
            "parse_agent_codes: markdown mode — %d codes parsed (no confidence scores)",
            len(results),
        )

    return results


def score(
    case: GoldenCase,
    agent_output: dict,
    latency_seconds: Optional[float] = None,
) -> EvalResult:
    """
    Score a single agent response against a golden case.

    Args:
        case          — the golden case with expected codes
        agent_output  — the parsed ClaimAuditLog JSON (or markdown-parsed dict)
        latency_seconds — optional wall-clock time for the agent call

    Returns:
        EvalResult with F1, precision, recall, and detailed breakdown
    """
    result = EvalResult(
        case_id=case.case_id,
        patient_id=case.patient_id,
        payer=case.payer,
        overall_status=agent_output.get("overall_status"),
        overall_confidence=agent_output.get("overall_confidence"),
        latency_seconds=latency_seconds,
        agent_claim_id=agent_output.get("claim_id"),
    )

    # Parse predicted codes
    predicted_raw = parse_agent_codes(agent_output)
    predicted_set = {(cpt, mod) for cpt, mod, *_ in predicted_raw}
    predicted_map = {
        (cpt, mod): (conf, status, citation)
        for cpt, mod, conf, status, citation in predicted_raw
    }

    # Ground truth set
    expected_set = case.expected_set

    # Resolve alternative groups — if any predicted code matches an alternative,
    # treat it as the canonical TP and drop the other group members from expected_set
    # so they don't generate false negatives.
    resolved_expected = set(expected_set)
    for group in case.alternative_groups:
        group_set = set(group)
        predicted_from_group = group_set & predicted_set
        if predicted_from_group:
            # One or more alternatives predicted — remove the unpredicted ones
            resolved_expected -= (group_set - predicted_from_group)
        else:
            # Nothing predicted from this group — keep only the first as the FN
            resolved_expected -= (group_set - {group[0]})
    expected_set = resolved_expected

    # True positives
    for pair in predicted_set & expected_set:
        conf, status, citation = predicted_map.get(pair, (None, None, None))
        result.true_positives.append(CodeResult(
            cpt=pair[0],
            modifier=pair[1],
            outcome="true_positive",
            confidence=conf,
            status=status,
            citation=citation,
        ))

    # False positives (predicted but not expected)
    for pair in predicted_set - expected_set:
        conf, status, citation = predicted_map.get(pair, (None, None, None))
        result.false_positives.append(CodeResult(
            cpt=pair[0],
            modifier=pair[1],
            outcome="false_positive",
            confidence=conf,
            status=status,
            citation=citation,
        ))

    # False negatives (expected but not predicted)
    for pair in expected_set - predicted_set:
        result.false_negatives.append(CodeResult(
            cpt=pair[0],
            modifier=pair[1],
            outcome="false_negative",
        ))

    # Must-not-include violations (hallucination check)
    predicted_cpts = {cpt for cpt, _ in predicted_set}
    for banned_cpt in case.must_not_include:
        if banned_cpt in predicted_cpts:
            pairs = [(c, m) for c, m in predicted_set if c == banned_cpt]
            for cpt, mod in pairs:
                conf, _, _ = predicted_map.get((cpt, mod), (None, None, None))
                result.must_not_include_violations.append(
                    MustNotIncludeViolation(cpt=cpt, modifier=mod, confidence=conf)
                )

    # Required modifier pair check
    for req_cpt, req_mod in case.required_modifier_pairs:
        if (req_cpt, req_mod) not in predicted_set:
            result.missing_modifiers.append((req_cpt, req_mod))

    return result


# ---------------------------------------------------------------------------
# Aggregate reporter
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    """Aggregate report across all eval cases."""
    results: list[EvalResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def macro_f1(self) -> float:
        """Average F1 across all cases."""
        if not self.results:
            return 0.0
        return sum(r.f1 for r in self.results) / len(self.results)

    @property
    def macro_precision(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.precision for r in self.results) / len(self.results)

    @property
    def macro_recall(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.recall for r in self.results) / len(self.results)

    @property
    def hallucination_rate(self) -> float:
        """Fraction of cases with at least one must-not-include violation."""
        if not self.results:
            return 0.0
        return sum(
            1 for r in self.results if r.must_not_include_violations
        ) / len(self.results)

    @property
    def modifier_accuracy(self) -> float:
        """Fraction of required modifier pairs correctly applied."""
        total_required = sum(
            len(r.missing_modifiers) +
            sum(1 for tp in r.true_positives if tp.modifier is not None)
            for r in self.results
        )
        total_correct = sum(
            sum(1 for tp in r.true_positives if tp.modifier is not None)
            for r in self.results
        )
        return total_correct / total_required if total_required > 0 else 1.0

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("EVAL REPORT")
        print("=" * 60)
        print(f"Cases:          {self.total}")
        print(f"Passed:         {self.passed} ({100*self.passed/self.total:.0f}%)")
        print(f"Failed:         {self.failed}")
        print(f"Macro F1:       {self.macro_f1:.3f}")
        print(f"Macro Precision:{self.macro_precision:.3f}")
        print(f"Macro Recall:   {self.macro_recall:.3f}")
        print(f"Modifier Acc:   {self.modifier_accuracy:.3f}")
        print(f"Hallucination:  {self.hallucination_rate:.1%}")
        print("-" * 60)
        for r in self.results:
            print(f"  {r.summary_line()}")
            if r.true_positives:
                print(f"    Correct: {', '.join(str(c) for c in r.true_positives)}")
            if r.false_negatives:
                print(f"    Missed:  {', '.join(str(c) for c in r.false_negatives)}")
            if r.false_positives:
                print(f"    Extra:   {', '.join(str(c) for c in r.false_positives)}")
            if r.missing_modifiers:
                mods = ", ".join(f"{c}-{m}" for c, m in r.missing_modifiers)
                print(f"    Missing modifiers: {mods}")
            if r.must_not_include_violations:
                halluc = ", ".join(v.cpt for v in r.must_not_include_violations)
                print(f"    Hallucinations: {halluc}")
        print("=" * 60)