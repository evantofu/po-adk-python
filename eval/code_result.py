"""
code_result.py  —  po-adk-python/eval/code_result.py

Drop-in replacement / patch for CodeResult with clean repr.

Before:  CodeResult(code='96127', modifier='59', source='agent', ...)
After:   96127-59   (or just 96127 when no modifier)

Also shows how to patch the print calls in runner.py if you'd rather
not move the class definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
#  CodeResult with clean __repr__ / __str__                                    #
# --------------------------------------------------------------------------- #

@dataclass
class CodeResult:
    """A single CPT code emitted by the agent."""
    code: str
    modifier: Optional[str] = None
    source: str = "agent"           # "agent" | "golden"
    confidence: Optional[float] = None
    extra: dict = field(default_factory=dict)

    # ---- display ---------------------------------------------------------- #

    @property
    def display(self) -> str:
        """Short human-readable label: '96127' or '96127-59'."""
        return f"{self.code}-{self.modifier}" if self.modifier else self.code

    def __repr__(self) -> str:
        return self.display

    def __str__(self) -> str:
        return self.display

    # ---- equality (used in F1 matching) ----------------------------------- #

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CodeResult):
            return self.code == other.code and self.modifier == other.modifier
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.code, self.modifier))


# --------------------------------------------------------------------------- #
#  Minimal patch for runner.py — if you can't refactor the class itself       #
#                                                                               #
#  In runner.py, change:                                                       #
#    print(f"  Missed:  {missed_codes}")                                       #
#  to:                                                                         #
#    print(f"  Missed:  {_fmt(missed_codes)}")                                 #
#                                                                               #
#  and add this helper:                                                        #
# --------------------------------------------------------------------------- #

def _fmt(codes) -> str:
    """
    Format a collection of CodeResult (or plain strings) compactly.
    Works whether codes is a list, set, or single item.

    Examples:
        _fmt([CodeResult('96127','59'), CodeResult('99395')])
        => '96127-59, 99395'
    """
    if not hasattr(codes, '__iter__') or isinstance(codes, str):
        codes = [codes]

    parts = []
    for c in codes:
        if hasattr(c, 'display'):
            parts.append(c.display)
        elif hasattr(c, 'code'):          # duck-typed CodeResult without display
            mod = getattr(c, 'modifier', None)
            parts.append(f"{c.code}-{mod}" if mod else c.code)
        else:
            parts.append(str(c))
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
#  Example — how verbose output looks after the fix                            #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    codes = [
        CodeResult("99395"),
        CodeResult("96127", modifier="59"),
        CodeResult("Z23",   modifier=None),
    ]

    print("=== display examples ===")
    for c in codes:
        print(f"  repr:    {c!r}")
        print(f"  str:     {c!s}")
        print(f"  display: {c.display}")

    print()
    print("=== in a list (old vs new) ===")
    print(f"  old: CodeResult(code='96127', modifier='59', ...)")
    print(f"  new: {codes}")

    print()
    print("=== _fmt helper ===")
    print(f"  {_fmt(codes)}")
    print(f"  {_fmt(codes[1])}")   # single item