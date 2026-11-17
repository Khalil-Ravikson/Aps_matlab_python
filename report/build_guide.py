r"""Guia de estudo em PDF: Parte I (do impulso ao filtro) e Parte II (lista de resoluções).

Reaproveita o pipeline do relatório (Markdown + MathML → Chromium). Os textos da
Parte I estão em ``content/tutorial.py``; as resoluções da Parte II usam os mesmos
passos validados de ``content/<exercício>.py``.

Uso (na raiz do projeto)::

    python -m report.build_guide               # gera report/out/guia_de_estudo.pdf
    python -m report.build_guide --resolucoes  # gera report/out/lista_de_resolucoes.pdf

Os PDFs são entregáveis acadêmicos e não são versionados (ver .gitignore).
"""

from __future__ import annotations

import argparse
import html
import re
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

from content import registry, tutorial
from content.errata import items as errata_items
from content.model import fill
from dsp_av2 import plotting
from dsp_av2.fir import analysis as fa
from dsp_av2.fir import ideal, windows as W

from .build_report import (CSS, TOC_END, locate, math, md, mod_ex, plt, print_pdf, status, svg, sym,
                           values_table)

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "report" / "out" / "guia_de_estudo.pdf"

EXTRA_CSS = """
.answer { border: 1.2pt solid #0f7a5a; border-radius: 3pt; padding: 6pt 10pt; margin: 10pt 0; background: #f2faf6;
          break-inside: avoid; }
.answer-h { font-family: "Liberation Sans", sans-serif; font-weight: bold; color: #0f7a5a; font-size: 9.5pt;
            letter-spacing: .05em; margin-bottom: 3pt; }
.part { break-before: page; height: 230mm; display: flex; flex-direction: column; justify-content: center; }
.part-n { font-family: "Liberation Sans", sans-serif; color: #5a5953; letter-spacing: .1em; font-size: 11pt; }
.part-t { font-family: "Liberation Sans", sans-serif; font-size: 28pt; font-weight: bold; color: #14304f; margin: 6pt 0; }
.part-d { font-size: 11.5pt; color: #333; max-width: 140mm; }
.stmt-small { font-size: 9pt; }
.step-compact .step { margin: 5pt 0; padding: 4pt 8pt; }
.tsec h1 { margin-top: 18pt; padding-top: 6pt; border-top: 1.5pt solid #14304f; }
.tsec figure { margin-top: 6pt; }
"""

# Figuras da Parte I (dados calculados aqui, desenhados com o estilo do projeto)


def fig_moving_average() -> str:
    w = np.linspace(0, np.pi, 1001)
    H = np.abs(fa.frequency_response(np.array([0.5, 0.5]), w))
    fig, ax = plt.subplots(figsize=(6.5, 2.6), constrained_layout=True)
    ax.plot(w / np.pi, H, color=plotting.SERIES, lw=1.6)
    ax.set_xlabel("ω / π")
    ax.set_ylabel("|H(e^jω)|")
    ax.set_ylim(0, 1.05)
    plotting._style(ax)
    return svg(fig)


def fig_windows() -> str:
    M, wc = 31, 0.5 * np.pi
    w = np.linspace(0, np.pi, 4001)
    fig, ax = plt.subplots(figsize=(7.0, 3.0), constrained_layout=True)
    for name, win, color in (("retangular (corte seco)", np.ones(M), "#b3261e"), ("Hann", W.hann(M), plotting.SERIES),
                             ("Blackman", W.blackman(M), "#1baf7a")):
        h = ideal.ideal_lowpass(wc, M) * win
        with np.errstate(divide="ignore"):
            db = 20 * np.log10(np.abs(fa.frequency_response(h, w)))
        ax.plot(w / np.pi, db, color=color, lw=1.3, label=name)
    ax.set_ylim(-100, 5)
    ax.set_xlabel("ω / π")
    ax.set_ylabel("|H| (dB)")
    ax.legend(frameon=False, fontsize=8)
    plotting._style(ax)
    return svg(fig)


TUTORIAL_FIGS = {  # figura inserida ao final da seção de mesmo índice
    1: ("A média móvel de 2 pontos é um passa-baixas: ganho 1 em ω = 0 e ganho 0 em ω = π.", fig_moving_average),
    5: ("Mesmo passa-baixas ideal (M = 31, corte em 0.5π) cortado por três janelas. O corte seco (retangular) deixa "
        "lóbulos altos; Hann e Blackman descem muito mais, ao custo de uma transição mais larga.", fig_windows),
    8: ("P7.9 resolvido: resposta ao impulso, magnitude em dB e zoom na passagem.",
        lambda: svg(mod_ex("p7_09").plot(registry.load("p7_09").objects()["final"]))),
}


def part_one(toc: list) -> str:
    ctx = tutorial.context()
    out = ['<section class="part"><div class="part-n">PARTE I</div><div class="part-t">Do impulso ao filtro</div>'
           '<div class="part-d">Um curso curto, para quem já sabe o que é resposta ao impulso: da resposta em frequência '
           'até resolver uma questão de projeto de filtro do começo ao fim.</div></section>']
    for i, (title, body) in enumerate(tutorial.SECTIONS):
        toc.append((1, title))
        sec = f'<section class="tsec"><h1 id="t{len(toc)}">{html.escape(title)}</h1>{md(fill(body, ctx))}'
        if i in TUTORIAL_FIGS:
            cap, fn = TUTORIAL_FIGS[i]
            sec += f"<figure>{fn()}<figcaption>{html.escape(cap)}</figcaption></figure>"
        out.append(sec + "</section>")
    return "".join(out)


def resolution(ex_id: str, toc: list) -> str:
    mod = registry.load(ex_id)
    C, objs, ctx = mod.CONTENT, mod.objects(), mod.context()
    spec = objs["spec"]
    rep = objs.get("final_report") or objs.get("report")
    title = f"{C.number}: {C.title}"
    toc.append((1, title))
    h = [f'<section class="chapter ch{C.chapter}"><div class="chap-tag">Capítulo {C.chapter} · '
         f'{"FIR" if C.chapter == 7 else "IIR"}</div><h1 id="t{len(toc)}">{html.escape(title)}</h1>',
         f'<blockquote class="statement stmt-small">{html.escape(spec.statement)}</blockquote>',
         "<h2>O que se pede</h2>" + md(fill(C.asked, ctx)),
         "<h2>Dados</h2>" + values_table(C.data_rows, ctx),
         '<h2>Resolução</h2><div class="step-compact">']
    for k, s in enumerate(C.steps, 1):
        h.append(f'<div class="step"><div class="step-h"><span>PASSO {k}</span> {md(fill(s.title, ctx))[3:-4]}</div>'
                 + md(fill(s.reasoning, ctx))
                 + (f'<div class="eq">{math(fill(s.equation, ctx), True)}</div>' if s.equation else "")
                 + (f'<div class="note">{md(fill(s.note, ctx))}</div>' if s.note else "") + "</div>")
    h.append("</div>")
    rows = "".join(f"<tr><td>{sym(fill(v.symbol, ctx))}</td><td><b>{md(fill(v.value, ctx))[3:-4]}</b></td>"
                   f"<td>{md(fill(v.meaning, ctx))[3:-4]}</td></tr>" for v in C.highlights)
    h.append(f'<div class="answer"><div class="answer-h">RESPOSTA</div><table class="vals">{rows}</table>'
             f"<p>Verificação: {status(rep.passed)} em todos os critérios do enunciado.</p>"
             f"{md(fill(C.conclusion, ctx))}</div>")
    errs = [errata_items()[e] for e in C.errata if e.startswith("E")]
    for e in errs:
        h.append(f'<div class="errata"><b>Atenção ({e.id}):</b> o documento de referência afirma {md(e.claim)[3:-4]}. '
                 f"<b>Correto:</b> {md(e.correction)[3:-4]}</div>")
    h.append(f'<figure>{svg(mod_ex(ex_id).plot(objs.get("final") or objs["proto"]))}'
             f"<figcaption>Gráficos pedidos pelo enunciado ({C.number}).</figcaption></figure></section>")
    return "\n".join(h)


def document(only_resolutions: bool = False) -> tuple[str, list]:
    toc: list = []
    if only_resolutions:
        cover = f"""
<section class="cover">
  <div class="cv-top">Processamento Digital de Sinais · Lista de resoluções</div>
  <div class="cv-title">Lista de resoluções</div>
  <div class="cv-sub">Exercícios P7.9–P7.16 (filtros FIR) e P8.12–P8.15 (protótipos IIR) de Ingle &amp; Proakis,
  Digital Signal Processing Using MATLAB, 3. ed.</div>
  <div class="cv-scope">Para cada questão: enunciado, o que se pede, dados, resolução passo a passo, resposta e gráficos</div>
  <div class="cv-date">Gerado em {date.today().strftime('%d/%m/%Y')} a partir do código e do conteúdo validado do projeto DSP Filter Lab</div>
</section>"""
        body = [resolution(e.id, toc) for e in registry.EXERCISES]
        return _wrap(cover, body, toc, "Lista de resoluções — DSP"), toc
    cover = f"""
<section class="cover">
  <div class="cv-top">Processamento Digital de Sinais · Guia de estudo</div>
  <div class="cv-title">Do impulso ao filtro</div>
  <div class="cv-sub">Curso curto de projeto de filtros digitais e lista com as resoluções dos exercícios
  P7.9–P7.16 e P8.12–P8.15 (Ingle &amp; Proakis, 3. ed.)</div>
  <div class="cv-scope">Parte I: teoria do zero, com exemplos resolvidos · Parte II: as 12 resoluções passo a passo</div>
  <div class="cv-date">Gerado em {date.today().strftime('%d/%m/%Y')} a partir do código e do conteúdo validado do projeto DSP Filter Lab</div>
</section>"""
    body = [part_one(toc)]
    body.append('<section class="part"><div class="part-n">PARTE II</div><div class="part-t">Lista de resoluções</div>'
                '<div class="part-d">As 12 questões, cada uma com o que se pede, os dados, a resolução passo a passo, '
                'a resposta destacada e os gráficos pedidos.</div></section>')
    for e in registry.EXERCISES:
        body.append(resolution(e.id, toc))
    return _wrap(cover, body, toc, "Guia de estudo — DSP"), toc


def _wrap(cover: str, body: list, toc: list, title: str) -> str:
    rows = "".join(f'<div class="toc-l{lvl}"><a href="#t{i}">{html.escape(t)}</a><span class="dots"></span>'
                   f'<span class="pg">\x01{i}\x01</span></div>' for i, (lvl, t) in enumerate(toc, 1))
    toc_html = (f'<section class="toc"><h1 class="toc-h">Sumário</h1>{rows}'
                f'<div class="toc-end">{TOC_END}</div></section>')
    return (f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>{title}</title>"
            f"<style>{CSS}{EXTRA_CSS}</style></head><body>{cover}{toc_html}{''.join(body)}</body></html>")


def build(out: Path = OUT_DEFAULT, only_resolutions: bool = False) -> Path:
    html_doc, toc = document(only_resolutions)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "g.html").write_text(re.sub("\x01\\d+\x01", "000", html_doc), encoding="utf-8")
        print_pdf(tmp / "g.html", tmp / "pass1.pdf")
        pages = locate(tmp / "pass1.pdf", [t for _, t in toc])
        final = re.sub("\x01(\\d+)\x01", lambda m: str(pages.get(toc[int(m.group(1)) - 1][1], "")), html_doc)
        (tmp / "g2.html").write_text(final, encoding="utf-8")
        out.parent.mkdir(parents=True, exist_ok=True)
        print_pdf(tmp / "g2.html", out)
    missing = [t for _, t in toc if t not in pages]
    if missing:
        print("aviso: títulos não localizados no PDF:", missing)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resolucoes", action="store_true", help="apenas a lista das 12 resoluções")
    args = ap.parse_args()
    out = args.out or (ROOT / "report" / "out" / "lista_de_resolucoes.pdf" if args.resolucoes else OUT_DEFAULT)
    print("gerado:", build(out, args.resolucoes))


if __name__ == "__main__":
    main()
