#!/usr/bin/env python3
"""
Generate fake pipeline outputs (Step1..Step4) in `data/out-fake/`
mimicking the structure in `data/out/<company>/<docset>/`.

Privacy-safe: generates only demo companies (companyA/B/C) with generic filenames.

Outputs:
- data/out-fake/ingest.jsonl
- data/out-fake/<company>/<docset>/sections.jsonl
- data/out-fake/<company>/<docset>/chunks.jsonl
- data/out-fake/<company>/<docset>/retrieval.jsonl
- data/out-fake/<company>/<docset>/extraction.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------
# Helpers
# -----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def clamp_quote_25_words(text: str) -> str:
    words = text.replace("\n", " ").split()
    return " ".join(words[:25])


# -----------------------------
# Fake content templates
# -----------------------------

CLAUSE_FAMILIES = ["veto", "liquidity", "exit", "anti_dilution"]

DEFAULT_QUERIES = {
    "veto": [
        "reserved matters",
        "décisions réservées",
        "investor consent",
        "accord préalable investisseurs",
        "protective provisions",
    ],
    "liquidity": [
        "transfer restriction",
        "cession de titres",
        "ROFR",
        "droit de préemption",
        "approval / agrément",
        "lock-up",
    ],
    "exit": [
        "drag along",
        "tag along",
        "sale of the company",
        "sortie conjointe",
        "cession forcée",
    ],
    "anti_dilution": [
        "anti-dilution",
        "ratchet",
        "weighted average",
        "full ratchet",
        "conversion price",
        "down round",
        "BSA",
    ],
}

SHA_SNIPPETS = {
    "liquidity": [
        "Any transfer of Shares is subject to a right of first refusal (ROFR) in favor of the Investors, exercisable within 20 Business Days.",
        "Toute cession de titres à un tiers est soumise à agrément préalable, sous réserve des transferts autorisés.",
    ],
    "exit": [
        "Holders representing at least 70% of the voting rights may require all shareholders to sell their Shares in a Sale of the Company (drag-along).",
        "Les actionnaires minoritaires bénéficient d'un droit de sortie conjointe (tag-along) aux mêmes conditions.",
    ],
    "veto": [
        "The Company shall not, without Investor Majority consent, approve the Annual Budget or incur Indebtedness above $2,000,000.",
        "Sont soumises à l'accord préalable des Investisseurs : émission de titres, endettement au-delà de 1 000 000€, modification statutaire.",
    ],
    # Often absent in SHA; we may include 0-1 related chunks depending on RNG
    "anti_dilution": [
        "An adjustment mechanism may apply to certain subscription warrants (ratchet) if new securities are issued at a lower price, subject to conditions.",
    ],
}

TS_SNIPPETS = {
    "anti_dilution": [
        "Broad-based weighted average anti-dilution adjustment applies upon a Down Round; the conversion price shall be adjusted accordingly.",
        "Full ratchet protection applies if shares are issued below the Original Issue Price, subject to customary carve-outs.",
    ],
    "exit": [
        "Drag-along threshold set at 70% of Ordinary Shares in a Sale of the Company; all holders must sell on the same terms.",
    ],
    "liquidity": [
        "Transfer restrictions include ROFR in favor of Investors with a 20 Business Day exercise period and standard permitted transfers.",
    ],
    "veto": [
        "Protective provisions require Investor Majority consent for incurring debt above $2,000,000 and approving the Annual Budget.",
    ],
}


@dataclass
class DocsetSpec:
    company: str
    docset_folder: str   # folder name under company
    source_file: str     # used in evidence.source_file
    doc_type: str        # "SHA" or "TS"
    language: str        # "fr" or "en"
    n_pages: int


def make_specs() -> List[DocsetSpec]:
    """
    Demo-safe specs only (companyA/B/C), with TS + SHA.
    companyB has a SHA amendment to demonstrate conflicts.
    """
    return [
        # companyA
        DocsetSpec(
            company="companyA",
            docset_folder="CompanyA_-_SHA_v1_signed",
            source_file="CompanyA_Shareholders_Agreement_v1_signed.pdf",
            doc_type="SHA",
            language="en",
            n_pages=55,
        ),
        DocsetSpec(
            company="companyA",
            docset_folder="CompanyA_-_Term_Sheet_-_2024-06",
            source_file="CompanyA_Term_Sheet_2024-06.pdf",
            doc_type="TS",
            language="en",
            n_pages=9,
        ),

        # companyB (SHA + amendment + TS)
        DocsetSpec(
            company="companyB",
            docset_folder="CompanyB_-_SHA_v3_signed",
            source_file="CompanyB_Shareholders_Agreement_v3_signed.pdf",
            doc_type="SHA",
            language="fr",
            n_pages=72,
        ),
        DocsetSpec(
            company="companyB",
            docset_folder="CompanyB_-_SHA_Amendment_2025-02",
            source_file="CompanyB_SHA_Amendment_2025-02.pdf",
            doc_type="SHA",
            language="fr",
            n_pages=5,
        ),
        DocsetSpec(
            company="companyB",
            docset_folder="CompanyB_-_Term_Sheet_-_2025-01",
            source_file="CompanyB_Term_Sheet_2025-01.pdf",
            doc_type="TS",
            language="en",
            n_pages=8,
        ),

        # companyC
        DocsetSpec(
            company="companyC",
            docset_folder="CompanyC_-_SHA_v2_signed",
            source_file="CompanyC_Shareholders_Agreement_v2_signed.pdf",
            doc_type="SHA",
            language="en",
            n_pages=63,
        ),
        DocsetSpec(
            company="companyC",
            docset_folder="CompanyC_-_Term_Sheet_-_2023-11",
            source_file="CompanyC_Term_Sheet_2023-11.pdf",
            doc_type="TS",
            language="en",
            n_pages=10,
        ),
    ]


# -----------------------------
# Generators per step
# -----------------------------

def gen_sections(spec: DocsetSpec, rng: random.Random) -> List[Dict[str, Any]]:
    """
    sections.jsonl: one line per section block.
    Robust for small docs (e.g., term sheets with <10 pages).
    """
    # You need at least 1 section, and at most one section per page (practically)
    max_sections = max(1, spec.n_pages)  # n_pages >= 1
    # We need (n_sections - 1) breakpoints sampled from range(1, n_pages)
    # So n_sections cannot exceed n_pages (otherwise k > population).
    n_sections = rng.randint(6, 9)
    n_sections = min(n_sections, max_sections)

    # If n_pages == 1, there is no valid breakpoint range; single section spanning page 1
    if spec.n_pages <= 1 or n_sections <= 1:
        return [{
            "company_id": spec.company,
            "doc_id": f"{spec.company}::{spec.docset_folder}",
            "source_file": spec.source_file,
            "section_title": "Document",
            "page_start": 1,
            "page_end": spec.n_pages,
        }]

    # Population size is (n_pages - 1). Clamp k accordingly.
    population = list(range(1, spec.n_pages))  # 1..n_pages-1
    k = min(n_sections - 1, len(population))
    breaks = sorted(rng.sample(population, k=k))

    pages = [1] + breaks + [spec.n_pages + 1]

    titles_fr = [
        "Préambule", "Définitions", "Gouvernance", "Cessions de Titres", "Décisions réservées",
        "Sortie / Liquidité", "Dispositions diverses", "Annexes"
    ]
    titles_en = [
        "Preamble", "Definitions", "Governance", "Transfer Restrictions", "Protective Provisions",
        "Exit / Liquidity", "Miscellaneous", "Schedules"
    ]
    titles = titles_fr if spec.language == "fr" else titles_en

    rows = []
    # pages length is k+2, so sections count is len(pages)-1
    for i in range(len(pages) - 1):
        p_start = pages[i]
        p_end = pages[i + 1] - 1
        rows.append({
            "company_id": spec.company,
            "doc_id": f"{spec.company}::{spec.docset_folder}",
            "source_file": spec.source_file,
            "section_title": titles[i % len(titles)],
            "page_start": p_start,
            "page_end": p_end,
        })
    return rows

def gen_chunks(spec: DocsetSpec, rng: random.Random) -> List[Dict[str, Any]]:
    doc_id = f"{spec.company}::{spec.docset_folder}"
    rows: List[Dict[str, Any]] = []

    snippet_pool = SHA_SNIPPETS if spec.doc_type == "SHA" else TS_SNIPPETS

    n_chunks = rng.randint(18, 28) if spec.doc_type == "SHA" and spec.n_pages > 8 else rng.randint(8, 14)

    # Anti-dilution is often absent in SHA (simulate)
    include_ad = True
    if spec.doc_type == "SHA" and rng.random() < 0.6:
        include_ad = False

    family_weights = {
        "veto": 0.28,
        "liquidity": 0.28,
        "exit": 0.28,
        "anti_dilution": 0.16 if (spec.doc_type == "TS" or include_ad) else 0.0,
    }

    families: List[str] = []
    for fam, w in family_weights.items():
        families += [fam] * int(w * 100)
    if not families:
        families = ["veto", "liquidity", "exit"]

    for i in range(n_chunks):
        fam = rng.choice(families)
        fam_snips = snippet_pool.get(fam) or snippet_pool.get("liquidity") or []
        text = rng.choice(fam_snips) if fam_snips else "Clause text unavailable."

        if rng.random() < 0.35:
            text += " " + rng.choice([
                "The specific mechanics are set out in this section.",
                "Les modalités sont précisées au présent article.",
                "Subject to customary exceptions and permitted transfers."
            ])

        page_start = rng.randint(1, max(1, spec.n_pages - 1))
        page_end = min(spec.n_pages, page_start + rng.randint(0, 1))

        article = None
        if spec.language == "fr" and spec.doc_type == "SHA" and rng.random() < 0.8:
            article = str(rng.randint(8, 25))

        rows.append({
            "company_id": spec.company,
            "doc_id": doc_id,
            "chunk_id": f"{spec.company}_{spec.doc_type.lower()}_c{str(i+1).zfill(3)}",
            "source_file": spec.source_file,
            "page_start": page_start,
            "page_end": page_end,
            "section_title": f"{'Article' if spec.language=='fr' else 'Section'} {article or rng.randint(4,14)}",
            "article": article,
            "text": text,
            "span": {"start_char": rng.randint(1, 50000), "end_char": rng.randint(50001, 90000)},
            "meta": {"clause_hint": fam},
        })

    # Hard-code a conflict teaser: companyB SHA amendment changes drag threshold
    if spec.company == "companyB" and "Amendment" in spec.docset_folder:
        rows.append({
            "company_id": spec.company,
            "doc_id": doc_id,
            "chunk_id": f"{spec.company}_{spec.doc_type.lower()}_c999",
            "source_file": spec.source_file,
            "page_start": 2,
            "page_end": 2,
            "section_title": "Article 2 — Modification Drag-along",
            "article": "2",
            "text": "Le seuil de drag-along mentionné dans l'accord principal est modifié et porté à 75% des droits de vote.",
            "span": {"start_char": 900, "end_char": 1400},
            "meta": {"clause_hint": "exit"},
        })

    return rows


def gen_retrieval(spec: DocsetSpec, chunks: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    by_family: Dict[str, List[Dict[str, Any]]] = {f: [] for f in CLAUSE_FAMILIES}
    for c in chunks:
        hint = c.get("meta", {}).get("clause_hint")
        if hint in by_family:
            by_family[hint].append(c)

    for fam in CLAUSE_FAMILIES:
        k = 8 if spec.doc_type == "SHA" else 6
        candidates = by_family.get(fam, [])

        # Allow empty hits (esp. anti-dilution in SHA)
        if fam == "anti_dilution" and spec.doc_type == "SHA" and len(candidates) == 0:
            hits = []
        else:
            rng.shuffle(candidates)
            chosen = candidates[: min(k, len(candidates))]
            hits = [{
                "chunk_id": c["chunk_id"],
                "score": round(rng.uniform(0.55, 0.92), 2),
                "source_file": c["source_file"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "article": c.get("article"),
                "text": c["text"][:280],
            } for c in chosen]
            hits.sort(key=lambda x: x["score"], reverse=True)

        rows.append({
            "company_id": spec.company,
            "docset": spec.docset_folder,
            "clause_family": fam,
            "queries": DEFAULT_QUERIES[fam],
            "k": k,
            "hits": hits,
        })

    return rows


def gen_extraction(spec: DocsetSpec, retrieval_rows: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    def top_hit(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        hits = row.get("hits", [])
        return hits[0] if hits else None

    rows: List[Dict[str, Any]] = []

    for fam_row in retrieval_rows:
        fam = fam_row["clause_family"]
        hit = top_hit(fam_row)

        # Probability of being present depends on family + doc_type
        base_present = 0.75
        if fam == "anti_dilution" and spec.doc_type == "SHA":
            base_present = 0.15
        if fam == "anti_dilution" and spec.doc_type == "TS":
            base_present = 0.88

        present = (hit is not None) and (rng.random() < base_present)

        items: List[Dict[str, Any]] = []
        if present:
            quote = clamp_quote_25_words(hit["text"])
            page = hit.get("page_start")
            article = hit.get("article")

            clause_type = {
                "veto": rng.choice(["Reserved matters", "Protective provisions", "Décisions réservées"]),
                "liquidity": rng.choice(["ROFR", "Transfer restrictions", "Agrément"]),
                "exit": rng.choice(["Drag-along", "Tag-along"]),
                "anti_dilution": rng.choice(["Weighted average", "Full ratchet"]),
            }[fam]

            beneficiary = {
                "veto": "investor",
                "liquidity": rng.choice(["investor", "mixed", "company"]),
                "exit": "mixed",
                "anti_dilution": "investor",
            }[fam]

            thresholds = None
            if fam == "exit":
                thresholds = rng.choice(["60%", "66.67%", "70%", "75%"])
            elif fam == "veto":
                thresholds = rng.choice(["250 000€", "500 000€", "1 000 000€", "$2,000,000"])
            elif fam == "liquidity":
                thresholds = rng.choice([None, "20 Business Days", "30 jours"])

            trigger = {
                "veto": "Approval of budget / debt / issuance of securities",
                "liquidity": "Proposed transfer of shares",
                "exit": "Third-party sale offer / sale process",
                "anti_dilution": "Down round / issuance below prior price",
            }[fam]

            effect = {
                "veto": "Requires Investor consent",
                "liquidity": "Restricts transfers or grants ROFR/approval right",
                "exit": "Forces or enables sale participation under conditions",
                "anti_dilution": "Adjusts conversion price or economic terms",
            }[fam]

            confidence = round(rng.uniform(0.70, 0.90), 2)
            flags: List[str] = []
            if rng.random() < 0.12:
                flags.append("ambiguous")
                confidence = round(min(confidence, 0.69), 2)

            items.append({
                "item_id": f"{spec.company}_{spec.doc_type.lower()}_{fam}_it001",
                "present": True,
                "clause_type": clause_type,
                "beneficiary": beneficiary,
                "trigger": trigger,
                "effect": effect,
                "thresholds": thresholds,
                "evidence": {
                    "quote": quote,
                    "source_file": hit["source_file"],
                    "page": page,
                    "article": article,
                },
                "confidence": confidence,
                "flags": flags,
            })

        else:
            items.append({
                "item_id": f"{spec.company}_{spec.doc_type.lower()}_{fam}_it001",
                "present": False,
                "clause_type": f"{fam} (unspecified)",
                "beneficiary": "unclear",
                "trigger": "",
                "effect": "",
                "thresholds": None,
                "evidence": {"quote": "", "source_file": "", "page": None, "article": None},
                "confidence": round(rng.uniform(0.15, 0.40), 2),
                "flags": ["missing_evidence"],
            })

        rows.append({
            "company_id": spec.company,
            "docset": spec.docset_folder,
            "doc_type": spec.doc_type,
            "clause_family": fam,
            "items": items,
        })

    # Conflict flagging example: if companyB has both SHA and Amendment docsets, mark exit items as conflicting
    if spec.company == "companyB" and ("Amendment" in spec.docset_folder):
        for r in rows:
            if r["clause_family"] == "exit":
                for it in r["items"]:
                    if it["present"]:
                        it["flags"] = list(set(it.get("flags", []) + ["conflicting"]))
                        it["thresholds"] = "75% (amended)"

    if spec.company == "companyB" and ("SHA_v3" in spec.docset_folder):
        for r in rows:
            if r["clause_family"] == "exit":
                for it in r["items"]:
                    if it["present"]:
                        it["flags"] = list(set(it.get("flags", []) + ["conflicting"]))
                        it["thresholds"] = "70% (base)"

    return rows


# -----------------------------
# Ingest root (Step 1)
# -----------------------------

def gen_ingest_rows(specs: List[DocsetSpec]) -> List[Dict[str, Any]]:
    rows = []
    for s in specs:
        rows.append({
            "company_id": s.company,
            "doc_id": f"{s.company}::{s.docset_folder}",
            "source_file": s.source_file,
            "file_hash_sha256": sha256_hex(s.source_file),
            "doc_type": s.doc_type,
            "language": s.language,
            "n_pages": s.n_pages,
            "pages": [{"page": i, "chars": 2500 + (i % 7) * 300} for i in range(1, min(6, s.n_pages + 1))],
            "extraction_warnings": [],
            "created_at": now_iso(),
        })
    return rows


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="data/out-fake", help="Output root folder (default: data/out-fake)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--clean", action="store_true", help="Delete out-root before generating")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    if args.clean:
        safe_rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    specs = make_specs()

    # Step 1: ingest.jsonl at root
    write_jsonl(out_root / "ingest.jsonl", gen_ingest_rows(specs))

    # Steps 2-4 per docset folder
    for spec in specs:
        docset_dir = out_root / spec.company / spec.docset_folder
        docset_dir.mkdir(parents=True, exist_ok=True)

        sections = gen_sections(spec, rng)
        write_jsonl(docset_dir / "sections.jsonl", sections)

        chunks = gen_chunks(spec, rng)
        write_jsonl(docset_dir / "chunks.jsonl", chunks)

        retrieval = gen_retrieval(spec, chunks, rng)
        write_jsonl(docset_dir / "retrieval.jsonl", retrieval)

        extraction = gen_extraction(spec, retrieval, rng)
        write_jsonl(docset_dir / "extraction.jsonl", extraction)

    print(f"[OK] Wrote demo-safe fake dataset to: {out_root.resolve()}")
    print("[OK] Structure:")
    for c in ["companyA", "companyB", "companyC"]:
        print(f"  - {out_root / c}")


if __name__ == "__main__":
    main()
