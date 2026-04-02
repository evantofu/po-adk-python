"""
eval/golden_cases.py

Loads, validates, and provides access to golden test cases.

A golden case defines:
  - The patient and payer context to send to the agent
  - The expected CPT codes and modifiers (ground truth)
  - Codes that must NOT appear (hallucination detection)
  - Metadata for reporting and filtering
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CASES_DIR = Path(__file__).parent / "cases"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ExpectedCode:
    """A single expected CPT code + optional modifier pair."""
    cpt: str
    modifier: Optional[str] = None

    def as_tuple(self) -> tuple[str, Optional[str]]:
        return (self.cpt, self.modifier)

    def __hash__(self):
        return hash(self.as_tuple())

    def __eq__(self, other):
        if isinstance(other, ExpectedCode):
            return self.as_tuple() == other.as_tuple()
        return NotImplemented

    def __repr__(self):
        if self.modifier:
            return f"{self.cpt}-{self.modifier}"
        return self.cpt


@dataclass
class GoldenCase:
    """A single golden test case."""
    case_id: str
    description: str
    patient_id: str
    payer: str
    expected_codes: list[ExpectedCode]
    must_not_include: list[str]          # CPT codes that must not appear
    hard_stops_expected: list[str]       # CPT codes expected to be removed
    required_modifier_pairs: list[tuple[str, str]]  # [(cpt, modifier), ...]
    metadata: dict = field(default_factory=dict)

    @property
    def expected_set(self) -> set[tuple[str, Optional[str]]]:
        """Set of (cpt, modifier) tuples for F1 scoring."""
        return {c.as_tuple() for c in self.expected_codes}

    @property
    def encounter_type(self) -> str:
        return self.metadata.get("encounter_type", "unknown")

    @property
    def complexity(self) -> str:
        return self.metadata.get("complexity", "unknown")

    @property
    def key_challenges(self) -> list[str]:
        return self.metadata.get("key_challenges", [])


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_case(path: Path) -> GoldenCase:
    """Load and validate a single golden case from a JSON file."""
    with open(path) as f:
        raw = json.load(f)

    expected_codes = [
        ExpectedCode(
            cpt=c["cpt"],
            modifier=c.get("modifier")
        )
        for c in raw["expected"]["codes"]
    ]

    required_modifier_pairs = [
        tuple(pair)
        for pair in raw.get("scoring", {}).get("required_modifier_pairs", [])
    ]

    return GoldenCase(
        case_id=raw["case_id"],
        description=raw["description"],
        patient_id=raw["patient_id"],
        payer=raw["payer"],
        expected_codes=expected_codes,
        must_not_include=raw["expected"].get("must_not_include", []),
        hard_stops_expected=raw["expected"].get("hard_stops_expected", []),
        required_modifier_pairs=required_modifier_pairs,
        metadata=raw.get("metadata", {}),
    )


def load_all_cases() -> list[GoldenCase]:
    """Load all golden cases from the cases/ directory."""
    if not CASES_DIR.exists():
        raise FileNotFoundError(
            f"Cases directory not found: {CASES_DIR}\n"
            "Create eval/cases/ and add golden case JSON files."
        )

    case_files = sorted(CASES_DIR.glob("*.json"))
    if not case_files:
        raise FileNotFoundError(f"No JSON files found in {CASES_DIR}")

    cases = []
    for path in case_files:
        try:
            case = load_case(path)
            cases.append(case)
            logger.debug("Loaded case: %s", case.case_id)
        except Exception as e:
            logger.error("Failed to load case %s: %s", path.name, e)
            raise

    logger.info("Loaded %d golden cases", len(cases))
    return cases


def load_case_by_id(case_id: str) -> GoldenCase:
    """Load a specific golden case by ID."""
    path = CASES_DIR / f"{case_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Case not found: {path}")
    return load_case(path)