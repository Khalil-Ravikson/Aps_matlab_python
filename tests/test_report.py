"""Relatório: o HTML é gerado do mesmo conteúdo, sem marcadores crus e com equações em MathML."""

import re
import shutil

import pytest

from report import build_report as R


@pytest.fixture(scope="module")
def doc():
    html, toc = R.document(["p7_13", "p8_13"])
    return html, toc


def test_no_unfilled_markers_or_raw_latex(doc):
    html, _ = doc
    assert "[[" not in html
    text = re.sub(r"<math.*?</math>", "", html, flags=re.S)       # fora das fórmulas
    text = re.sub(r"<(style|svg)\b.*?</\1>", "", text, flags=re.S)
    assert "$" not in text, text[text.index("$") - 80: text.index("$") + 80]


def test_all_sections_in_toc(doc):
    _, toc = doc
    titles = [t for _, t in toc]
    for sec in R.SECTION_TITLES:
        assert any(t.endswith(sec) for t in titles), sec
    assert titles[-1].endswith("Referências")


def test_equations_and_figures_present(doc):
    html, _ = doc
    assert html.count("<math") > 100
    assert html.count("<svg") >= 4          # figuras vetoriais Matplotlib
    assert "Correção ao documento de referência (E1)" in html
    assert "Correção ao documento de referência (E6)" in html


def test_institutional_fields_are_not_invented(doc):
    html, _ = doc
    assert html.count("⟨a preencher⟩") >= 4


@pytest.mark.skipif(not shutil.which("chromium"), reason="Chromium ausente")
def test_pdf_builds_with_numbered_toc(tmp_path):
    from pypdf import PdfReader
    out = R.build(tmp_path / "r.pdf", ["p7_13"])
    pages = PdfReader(str(out)).pages
    assert len(pages) > 8
    toc_text = pages[1].extract_text()
    assert re.search(r"Introdução ao projeto de filtros digitais\s*\d+", toc_text)
