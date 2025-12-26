# schemas/right_item.py
from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator




ClauseFamily = Literal["veto", "liquidity", "exit", "anti_dilution"]
Beneficiary = Literal["investor", "founders", "company", "mixed", "unclear"]
Flag = Literal["ambiguous", "conflicting", "missing_evidence", "quote_too_long", "missing_source", "invalid_enum"]


class Evidence(BaseModel):
    quote: Optional[str] = None  # <= 25 words, direct excerpt
    source_file: Optional[str] = None
    page: Optional[int] = None
    article: Optional[str] = None


class RightItem(BaseModel):
    company_id: str
    clause_family: ClauseFamily
    clause_type: str

    present: bool = False
    beneficiary: Beneficiary = "unclear"
    trigger: str = ""
    effect: str = ""
    thresholds: Optional[str] = None

    evidence: Evidence = Field(default_factory=Evidence)
    confidence: float = 0.0
    flags: List[str] = Field(default_factory=list)

    # --- validators ---
    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        if v is None:
            return 0.0
        return max(0.0, min(1.0, float(v)))

    @field_validator("flags")
    @classmethod
    def uniq_flags(cls, v: List[str]) -> List[str]:
        # preserve order, unique
        seen = set()
        out = []
        for f in v or []:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out
