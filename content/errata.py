"""Leitura de docs/errata_dta.md (fonte única das correções ao DTA)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ERRATA_FILE = Path(__file__).resolve().parents[1] / "docs" / "errata_dta.md"


@dataclass(frozen=True)
class ErrataItem:
    id: str
    exercise: str
    claim: str
    correction: str
    evidence: str


@lru_cache(maxsize=1)
def items() -> dict[str, ErrataItem]:
    out = {}
    for line in ERRATA_FILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*([EV]\d+)\s*\|(.*)\|\s*$", line)
        if m:
            cells = [c.strip() for c in m.group(2).split("|")]
            out[m.group(1)] = ErrataItem(m.group(1), *cells[:4])
    return out
