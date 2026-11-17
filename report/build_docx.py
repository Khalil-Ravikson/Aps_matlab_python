r"""Versões DOCX (Word) do relatório, do guia de estudo e da lista de resoluções.

Usa o mesmo HTML dos PDFs, com duas adaptações: figuras em PNG (o Word não aceita
SVG embutido em HTML) e sumário gerado pelo próprio Word (campo de sumário).
As equações em MathML viram **equações nativas do Word** (editáveis), pela
conversão do pandoc embutido no pacote ``pypandoc_binary``.

Uso (na raiz do projeto)::

    python -m report.build_docx                      # os três documentos
    python -m report.build_docx relatorio guia lista # escolher quais

Os arquivos são gravados em report/out/. Como os PDFs, não são versionados.
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

from content import registry

from . import build_guide as G
from . import build_report as R

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report" / "out"
DOCS = {
    "relatorio": ("relatorio.docx", "Relatório — DSP Filter Lab"),
    "guia": ("guia_de_estudo.docx", "Guia de estudo — Do impulso ao filtro"),
    "lista": ("lista_de_resolucoes.docx", "Lista de resoluções"),
}


def _html(kind: str) -> str:
    if kind == "relatorio":
        html, _ = R.document([e.id for e in registry.EXERCISES])
    elif kind == "guia":
        html, _ = G.document()
    else:
        html, _ = G.document(only_resolutions=True)
    html = re.sub(r'<section class="toc">.*?</section>', "", html, flags=re.S)   # o Word gera o sumário
    html = re.sub(r"<style>.*?</style>", "", html, flags=re.S)                    # CSS de impressão não se aplica
    return html.replace("\x01", "")


def build(kind: str, out_dir: Path = OUT_DIR) -> Path:
    import pypandoc
    name, title = DOCS[kind]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    previous = R.FIGURE_FORMAT
    R.FIGURE_FORMAT = "png"
    try:
        html = _html(kind)
    finally:
        R.FIGURE_FORMAT = previous
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "doc.html"
        src.write_text(html, encoding="utf-8")
        pypandoc.convert_file(str(src), "docx", format="html", outputfile=str(out),
                              extra_args=["--toc", "--toc-depth=2", "--metadata", "toc-title=Sumário",
                                          "--metadata", f"title={title}"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("docs", nargs="*", metavar="DOC", help=f"quais gerar: {', '.join(DOCS)} (padrão: todos)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    unknown = [d for d in args.docs if d not in DOCS]
    if unknown:
        ap.error(f"documento desconhecido: {', '.join(unknown)} (opções: {', '.join(DOCS)})")
    for kind in args.docs or DOCS:
        print("gerado:", build(kind, args.out_dir))


if __name__ == "__main__":
    main()
