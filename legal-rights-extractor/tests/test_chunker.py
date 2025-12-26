import pytest

from lre.chunking.chunker import (
    DocumentPage,
    step2_chunking,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def run_chunking(text: str):
    pages = [
        DocumentPage(
            source_file="test.pdf",
            page=1,
            text=text,
        )
    ]
    sections, chunks = step2_chunking(
        pages=pages,
        company_id="testco",
        max_tokens=200,
        overlap_ratio=0.1,
    )
    return sections, chunks


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

def test_never_empty_output():
    """Chunking must never return empty sections or chunks."""
    sections, chunks = run_chunking("Some random legal text.")
    assert len(sections) >= 1
    assert len(chunks) >= 1


def test_toc_is_ignored():
    """TOC entries must not become section titles."""
    text = """
    TABLE DES MATIERES

    1. DEFINITIONS ........................................ 4
    2. DECLARATIONS DES PARTIES ............................ 4

    <<<PAGE_BREAK>>>

    1. DEFINITIONS
    Definitions content here.
    """
    sections, _ = run_chunking(text)
    titles = [s.title for s in sections]

    assert "1. DEFINITIONS ........................................ 4" not in titles
    assert "1. DEFINITIONS" in titles


def test_annex_summary_is_not_a_section():
    """Annex summary lines should not become sections."""
    text = """
    28. SIGNATURE

    <<<PAGE_BREAK>>>

    Annexe (A) Business Plan Initial
    Annexe 1 Définitions

    <<<PAGE_BREAK>>>

    Annexe (A)
    Real annex content starts here.
    """
    sections, _ = run_chunking(text)
    titles = [s.title for s in sections]

    assert "Annexe (A) Business Plan Initial" not in titles
    assert "Annexe (A)" in titles


def test_annex_list_items_are_not_sections():
    """Enumerated list items must remain content, not sections."""
    text = """
    Annexe 1

    1. Arrêté des comptes annuels et proposition d’affectation du résultat ;
    2. Révocation des mandataires sociaux de la Société ;
    3. Mise à jour du business plan.
    """
    sections, chunks = run_chunking(text)
    titles = [s.title for s in sections]

    assert "Annexe 1" in titles
    assert not any(t.startswith("1.") for t in titles)

    # But list content must still be present in chunks
    full_text = " ".join(c.text for c in chunks)
    assert "Arrêté des comptes annuels" in full_text
    assert "Mise à jour du business plan" in full_text
