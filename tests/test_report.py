"""Relatório: o HTML é gerado do mesmo conteúdo, sem marcadores crus e com equações em MathML."""

import re
import shutil

import pytest

from report import build_report as R


@pytest.fixture(scope="module")
def doc():
    html, toc = R.document(["p7_13", "p8_13"])
    html = html.replace("\x01", "")
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
    assert titles[-1].endswith("Referências e limitações")


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


def test_analysis_chapters_render_in_report():
    from content import analysis
    for blocks in (analysis.comparison_blocks(), analysis.validation_blocks(), analysis.methodology_blocks()):
        out = R.blocks_html(blocks, 15, [0])
        assert "[[" not in out
        text = re.sub(r"<math.*?</math>", "", out, flags=re.S)
        text = re.sub(r"<(style|svg)\b.*?</\1>", "", text, flags=re.S)
        assert "$" not in text


def test_study_guide_document_is_complete():
    from content import tutorial
    from content.model import fill
    from report import build_guide as G
    ctx = tutorial.context()
    for _, body in tutorial.SECTIONS:
        assert "[[" not in fill(body, ctx)
    html, toc = G.document()
    html = html.replace("\x01", "")
    assert "[[" not in html
    text = re.sub(r"<(math|style|svg)\b.*?</\1>", "", html, flags=re.S)
    assert "$" not in text
    assert len(toc) == len(tutorial.SECTIONS) + 12
    assert html.count('class="answer"') == 12


def test_docx_has_native_equations_and_figures(tmp_path):
    pytest.importorskip("pypandoc")
    import zipfile
    from report import build_docx
    out = build_docx.build("lista", tmp_path)
    z = zipfile.ZipFile(out)
    xml = z.read("word/document.xml").decode()
    assert xml.count("<m:oMath") > 100                                   # equações nativas do Word
    assert len([n for n in z.namelist() if n.startswith("word/media/")]) == 12   # um gráfico por exercício
    plain = re.sub(r"<[^>]+>", "", re.sub(r"<m:oMath.*?</m:oMath>", "", xml, flags=re.S))
    assert "[[" not in plain and "$" not in plain
