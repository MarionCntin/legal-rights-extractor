from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
import requests
from pydantic import BaseModel, Field, ValidationError, field_validator
import time
from datetime import datetime

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _log(msg: str) -> None:
    if os.getenv("LRE_VERBOSE", "0") == "1":
        print(f"[{_ts()}] {msg}", flush=True)

ClauseFamily = Literal["veto", "liquidity", "exit", "anti_dilution"]
Beneficiary = Literal["investor", "founders", "company", "mixed", "unclear"]
Flag = Literal["ambiguous", "conflicting", "missing_evidence"]

class Evidence(BaseModel):
    quote: str = Field(..., description="Direct excerpt from context (<= 25 words).")
    source_file: str
    page: Optional[int] = None
    article: Optional[str] = None

    @field_validator("quote")
    @classmethod
    def quote_max_25_words(cls, v: str) -> str:
        # word count (robust)
        words = re.findall(r"\S+", v.strip())
        if len(words) > 25:
            raise ValueError(f"Evidence quote too long: {len(words)} words (max 25).")
        return v.strip().replace("\n", " ")

class RightItem(BaseModel):
    company_id: str
    clause_family: ClauseFamily
    clause_type: str
    present: bool
    beneficiary: Beneficiary
    trigger: str
    effect: str
    thresholds: Optional[str] = None
    evidence: Evidence
    confidence: float = Field(..., ge=0.0, le=1.0)
    flags: List[Flag] = Field(default_factory=list)

    @field_validator("clause_type", "trigger", "effect")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()

class ExtractionResult(BaseModel):
    items: List[RightItem]

@dataclass
class OllamaConfig:
    model: str = "qwen2.5:3b-instruct"
    host: str = "127.0.0.1:11434"
    temperature: float = 0.0
    top_p: float = 1.0
    num_ctx: int = 4096
    timeout_s: int = 600

def _extract_first_json_object(text: str) -> Dict[str, Any]:
    """Extract first {...} JSON object from LLM output."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))

def _chunks_to_context(hits: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    """
    Concatenate retrieved hits with provenance headers.
    Hard-bound total context size for CPU inference.
    """
    out: List[str] = []
    total = 0

    for i, h in enumerate(hits):
        header = (
            f"[CHUNK {i}] "
            f"source_file={h.get('source_file') or h.get('file') or ''} "
            f"page_start={h.get('page_start') or h.get('page') or ''} "
            f"page_end={h.get('page_end') or h.get('page') or ''} "
            f"section_title={h.get('section_title') or ''} "
            f"article={h.get('article') or ''}\n"
        )
        body = (h.get("text") or "").strip()
        block = header + body + "\n\n"

        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                out.append(block[:remaining])
            break

        out.append(block)
        total += len(block)

    return "".join(out).strip()


def extract_with_ollama(
    company_id: str,
    clause_family: ClauseFamily,
    hits: List[Dict[str, Any]],
    cfg: Optional[OllamaConfig] = None,
    max_chars: int = 3000,
) -> ExtractionResult:
    cfg = cfg or OllamaConfig()
    host = os.getenv("OLLAMA_HOST", cfg.host)
    model = os.getenv("OLLAMA_MODEL", cfg.model)

    _log(f"[step4] START company={company_id} family={clause_family} hits={len(hits)} max_chars={max_chars}")

    context = _chunks_to_context(hits, max_chars=max_chars)

    if os.getenv("LRE_VERBOSE", "0") == "1":
        preview = context[:600].replace("\n", " ")
        _log(f"[step4] context_chars={len(context)} preview={preview!r}")

    prompt = _prompt(company_id, clause_family, context)

    url = f"http://{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "num_ctx": cfg.num_ctx,
            "num_predict": 512,
        },
    }

    _log(f"[step4] POST {url} (waiting...)")
    resp = requests.post(url, json=payload, timeout=cfg.timeout_s)
    resp.raise_for_status()

    raw = resp.json().get("response", "")
    data = _extract_first_json_object(raw)

    try:
        result = ExtractionResult.model_validate(data)
    except ValidationError as e:
        _log(f"[step4] FAIL validation family={clause_family}: {e}")
        tail = raw[-800:] if isinstance(raw, str) else str(raw)[-800:]
        _log(f"[step4] raw_tail={tail!r}")
        raise ValueError(f"Schema validation failed: {e}") from e

    _log(f"[step4] DONE family={clause_family} items={len(result.items)}")
    return result


def run_extract_from_retrieval_jsonl(
    company_id: str,
    retrieval_jsonl_path: str,
    out_jsonl_path: str,
    cfg: Optional[OllamaConfig] = None,
    max_chars: int = 3000,
) -> None:
    cfg = cfg or OllamaConfig()
    _log(f"[step4] FILE {retrieval_jsonl_path} -> {out_jsonl_path}")

    os.makedirs(os.path.dirname(out_jsonl_path) or ".", exist_ok=True)

    with open(retrieval_jsonl_path, "r", encoding="utf-8") as fin, open(
        out_jsonl_path, "w", encoding="utf-8"
    ) as fout:
        for idx, line in enumerate(fin, start=1):
            if not line.strip():
                continue

            r = json.loads(line)
            clause_family = r["clause_family"]
            hits = r.get("hits") or r.get("top_hits") or r.get("chunks") or []

            _log(f"[step4] line={idx} clause_family={clause_family} hits={len(hits)}")

            try:
                result = extract_with_ollama(
                    company_id=company_id,
                    clause_family=clause_family,
                    hits=hits,
                    cfg=cfg,
                    max_chars=max_chars,
                )

                fout.write(
                    json.dumps(
                        {
                            "company_id": company_id,
                            "clause_family": clause_family,
                            "items": [item.model_dump() for item in result.items],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fout.flush()

            except Exception as e:
                _log(f"[step4] FAIL clause_family={clause_family}: {type(e).__name__}: {e}")
                # keep going
                continue

    _log(f"[step4] FINISHED -> {out_jsonl_path}")

def _prompt(company_id: str, clause_family: ClauseFamily, context: str) -> str:
    return (
        "You extract investor rights from legal text.\n"
        "Return VALID JSON ONLY with this shape: {\"items\": [ ... ]}.\n"
        "\n"
        f"company_id: {company_id}\n"
        f"clause_family: {clause_family}\n"
        "\n"
        "Rules:\n"
        "1) Use ONLY CONTEXT.\n"
        "2) No guessing: if not explicit -> present=false and flags=[\"missing_evidence\"].\n"
        "3) Each item MUST include evidence with:\n"
        "   - quote: <=25 words, verbatim, fully contained in ONE chunk body\n"
        "   - source_file: exactly as in that chunk header\n"
        "   - page: use page_start from that chunk header\n"
        "   - article: if clearly present (header or text)\n"
        "4) If definition is unclear -> add \"ambiguous\".\n"
        "5) If conflicting versions -> output multiple items, add \"conflicting\".\n"
        "\n"
        "Item fields:\n"
        "company_id, clause_family, clause_type, present, beneficiary,"
        "trigger, effect, thresholds, evidence{quote,source_file,page,article}, confidence, flags.\n"
        "\n"
        "CONTEXT:\n"
        f"{context}"
    )




