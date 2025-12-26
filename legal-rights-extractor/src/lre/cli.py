# src/lre/cli.py
import json
import shutil
from pathlib import Path
import typer
import os

from lre.config import Settings
from lre.ingest.loaders import load_any
from lre.chunking import DocumentPage, step2_chunking, write_jsonl
from lre.retrieval.bm25_retriever import run_retrieval, infer_default_paths
from lre.pipeline.extract import run_extract_from_retrieval_jsonl
from lre.pipeline.validate import run_step5



app = typer.Typer(no_args_is_help=True)
ingest_app = typer.Typer(no_args_is_help=True)
app.add_typer(
    ingest_app,
    name="ingest",
    help=(
        "Step 1 (Ingestion & Text Extraction): Load PDF/DOCX/TXT files, "
        "extract raw text with page-level provenance, and write normalized JSONL "
        "for downstream processing."
    ),
)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt"}


def compute_company_id(path: Path, raw_dir: Path) -> str:
    """
    Company id strategy:
    - If files are organized as raw_dir/<CompanyName>/<file>, company_id = <CompanyName>
    - If files are directly in raw_dir, company_id = filename stem
    """
    if path.parent == raw_dir:
        return path.stem
    return path.parent.name


def copy_to_out_raw(path: Path, raw_dir: Path, out_dir: Path) -> None:
    """
    Copy input file into out_dir/raw/ preserving relative structure from raw_dir.
    Example:
      raw_dir = data/raw
      path    = data/raw/CompanyA/SHA.pdf
      out     = data/out/raw/CompanyA/SHA.pdf
    """
    rel = path.relative_to(raw_dir)
    dest = out_dir / "raw" / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)


@ingest_app.command("run")
def run(
    raw_dir: Path = typer.Option(None, help="Directory containing input documents."),
    out_dir: Path = typer.Option(None, help="Directory to write outputs."),
):
    """
    Step 1 (Ingestion & Text Extraction): Loads PDF/DOCX/TXT files, extracts text with page mapping,
    writes out_dir/ingest.jsonl, and produces per-document sections.jsonl + chunks.jsonl.
    Also copies raw files to out_dir/raw/ for auditability.
    """

    s = Settings()
    raw_dir = raw_dir or s.raw_dir
    out_dir = out_dir or s.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect supported files only
    paths = sorted(
        [
            p for p in raw_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        ]
    )
    if not paths:
        typer.echo(f"No supported files found in {raw_dir} (expected {sorted(SUPPORTED_SUFFIXES)})")
        raise typer.Exit(code=1)

    ingest_out = out_dir / "ingest.jsonl"

    n_ok, n_fail = 0, 0

    with ingest_out.open("w", encoding="utf-8") as f:
        for path in paths:
            try:
                # Keep a provenance copy
                copy_to_out_raw(path, raw_dir, out_dir)

                # ---- Step 1: Load + normalize ----
                doc = load_any(path)

                # Write normalized ingestion record (one doc per line)
                f.write(json.dumps(doc.model_dump(), ensure_ascii=False) + "\n")

                # ---- Step 2: Chunking ----
                source_file = Path(doc.source_path).name
                pages = [
                    DocumentPage(
                        source_file=source_file,
                        page=p.page_index + 1,  # convert 0-index -> 1-index
                        text=p.text,
                    )
                    for p in doc.pages
                ]

                company_id = compute_company_id(path, raw_dir)

                sections, chunks = step2_chunking(
                    pages=pages,
                    company_id=company_id,
                    doc_id=doc.doc_id,
                    max_tokens=1100,
                    overlap_ratio=0.12,
                )

                # Per-document outputs (clean + scalable)
                doc_out = out_dir / company_id / path.stem
                write_jsonl(doc_out / "sections.jsonl", sections)
                write_jsonl(doc_out / "chunks.jsonl", chunks)

                typer.echo(f"[OK] {path.name}: {len(sections)} sections, {len(chunks)} chunks -> {doc_out}")
                n_ok += 1

            except Exception as e:
                n_fail += 1
                typer.echo(f"[FAIL] {path.name}: {e}")

    typer.echo(f"Written: {ingest_out}")
    typer.echo(f"OK: {n_ok} | FAIL: {n_fail}")

@app.command("retrieve")
def retrieve(
out_dir: str = typer.Option(
        "data/out",
        help="Pipeline output directory",
    ),
    chunks: str = typer.Option(
        None,
        help="Path to chunks.jsonl (overrides out_dir)",
    ),
    out: str = typer.Option(
        None,
        help="Path to retrieval.jsonl (overrides out_dir)",
    ),
    model: str = typer.Option(
        "bm25-okapi",
        help="BM25 backend label (ignored by BM25)",
    ),
    top_k: int = typer.Option(
        12,
        help="Top-k chunks per clause family",
    ),
    batch_size: int = typer.Option(
        64,
        help="Unused for BM25 (kept for interface stability)",
    ),
    split_by_company: bool = typer.Option(
        True,
        help="Group by company_id inside a chunks file",
    ),
):
    """
    Step 2 & 3 (Chunking & Structural Parsing): lexical retrieval using BM25.
    """
    out_dir_path = Path(out_dir)

    # Case 1: explicit paths
    if chunks is not None:
        chunks_path = Path(chunks)
        out_path = Path(out) if out else chunks_path.parent / "retrieval.jsonl"
        run_retrieval(
            chunks_path=chunks_path,
            out_path=out_path,
            model_name=model,
            top_k=top_k,
            batch_size=batch_size,
            split_by_company=split_by_company,
        )
        typer.echo(f"[OK] retrieval written to: {out_path}")
        return

    # Case 2: default "flat" layout data/out/chunks.jsonl exists
    default_chunks, default_out = infer_default_paths(out_dir_path)
    if default_chunks.exists():
        run_retrieval(
            chunks_path=default_chunks,
            out_path=default_out,
            model_name=model,
            top_k=top_k,
            batch_size=batch_size,
            split_by_company=split_by_company,
        )
        typer.echo(f"[OK] retrieval written to: {default_out}")
        return

    # Case 3: per-company layout: search recursively
    candidates = sorted(out_dir_path.rglob("chunks.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No chunks.jsonl found under {out_dir_path}. "
            f"Run Step 2 first, or pass --chunks PATH explicitly."
        )

    for ch_path in candidates:
        out_path = ch_path.parent / "retrieval.jsonl"
        run_retrieval(
            chunks_path=ch_path,
            out_path=out_path,
            model_name=model,
            top_k=top_k,
            batch_size=batch_size,
            split_by_company=split_by_company,
        )
        typer.echo(f"[OK] retrieval written to: {out_path}")


@app.command("extract")
def extract_cmd(
    company_id: str = typer.Option(..., "--company-id", help="Company identifier (e.g., eurolysine)"),
    retrieval: str = typer.Option(..., "--retrieval", help="Path to retrieval.jsonl (Step 3 output)"),
    out: str = typer.Option(..., "--out", help="Path to write extraction.jsonl (Step 4 output)"),
    timeout_s: int = typer.Option(600, "--timeout-s", help="HTTP timeout (seconds) for Ollama"),
    num_ctx: int = typer.Option(3072, "--num-ctx", help="Ollama context window"),
    max_chars: int = typer.Option(10000, "--max-chars", help="Max chars of retrieved context sent to the model"),
):
    """
    Step 4 (Extraction – LLM over retrieved evidence): Reads retrieval.jsonl and writes extraction.jsonl (one line per clause_family).
    """
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # Pass knobs down by temporarily overriding config/env
    # Best: update OllamaConfig and _chunks_to_context to accept max_chars
    from lre.pipeline.extract import OllamaConfig, run_extract_from_retrieval_jsonl

    cfg = OllamaConfig(timeout_s=timeout_s, num_ctx=num_ctx)
    run_extract_from_retrieval_jsonl(
        company_id=company_id,
        retrieval_jsonl_path=retrieval,
        out_jsonl_path=out,
        cfg=cfg,
        max_chars=max_chars,
    )

    typer.echo(f"Written: {out}")

@app.command("validate")
def step5_validate(
    extraction: Path = typer.Option(
        ...,
        "--extraction",
        "-e",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to Step 4 extraction.jsonl",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory to write Step 5 outputs",
    ),
    min_conf: float = typer.Option(
        0.60,
        "--min-conf",
        help="Confidence threshold below which items are flagged ambiguous",
    ),
    max_quote_words: int = typer.Option(
        25,
        "--max-quote-words",
        help="Maximum words allowed in evidence.quote (present=true items)",
    ),
    conflict_thr: float = typer.Option(
        0.75,
        "--conflict-thr",
        help="Similarity threshold below which present items are flagged conflicting",
    ),
):
    """
    Step 5 — Validation & Consolidation : validated_items.jsonl ; portfolio_table.csv ; portfolio_table.xlsx
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    validated_path, csv_path, xlsx_path = run_step5(
        extraction_jsonl=extraction,
        out_dir=out_dir,
        min_conf_ambiguous=min_conf,
        max_quote_words=max_quote_words,
        conflict_similarity_threshold=conflict_thr,
    )

    typer.secho(f"[OK] validated: {validated_path}", fg=typer.colors.GREEN)
    typer.secho(f"[OK] table (csv): {csv_path}", fg=typer.colors.GREEN)
    typer.secho(f"[OK] table (xlsx): {xlsx_path}", fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()
