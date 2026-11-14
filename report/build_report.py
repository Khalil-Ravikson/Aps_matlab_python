r"""Relatório acadêmico em PDF a partir do mesmo conteúdo da aplicação.

Fluxo: conteúdo (``content/``) → HTML (Markdown + equações em MathML, sem
internet) → PDF pelo Chromium headless. O sumário recebe os números de página
numa segunda passagem: o PDF da primeira é lido com ``pypdf`` para localizar
cada título.

Uso (na raiz do projeto)::

    python -m report.build_report            # exercícios com conteúdo escrito
    python -m report.build_report --out caminho.pdf
"""

from __future__ import annotations

import argparse
import html
import io
import re
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import markdown as md_lib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from latex2mathml.converter import convert as tex2mml  # noqa: E402
from pypdf import PdfReader  # noqa: E402

from content import common, registry  # noqa: E402
from content.errata import items as errata_items  # noqa: E402
from content.model import KINDS, fill, table_cell  # noqa: E402
from dsp_av2 import plotting  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSS = (Path(__file__).with_name("report.css")).read_text(encoding="utf-8")
OUT_DEFAULT = ROOT / "report" / "out" / "relatorio.pdf"

KIND_COLORS = {"dado": "#2a78d6", "calculado": "#4a3aa7", "escolhido": "#b07800",
               "resultado": "#0f7a5a", "criterio": "#8a3b12"}
KIND_ICONS = {"dado": "■", "calculado": "∑", "escolhido": "◆", "resultado": "●", "criterio": "✓"}


# ── Markdown + LaTeX → HTML ─────────────────────────────────────────────────

_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
_INLINE = re.compile(r"(?<![\\$])\$(?!\$)(.+?)(?<![\\$])\$")


def math(latex: str, display: bool = False) -> str:
    return tex2mml(latex.strip(), display="block" if display else "inline")


def md(text: str) -> str:
    """Markdown com fórmulas $...$ e $$...$$ convertidas em MathML."""
    store: list[str] = []

    def keep(s: str) -> str:
        store.append(s)
        return f"\x00{len(store) - 1}\x00"

    text = _DISPLAY.sub(lambda m: keep(f'<div class="eq">{math(m.group(1), True)}</div>'), text)
    text = _INLINE.sub(lambda m: keep(math(m.group(1))), text)
    out = md_lib.markdown(text, extensions=["tables"])
    return re.sub(r"\x00(\d+)\x00", lambda m: store[int(m.group(1))], out)


def badge(kind: str) -> str:
    c = KIND_COLORS[kind]
    return f'<span class="kind" style="border-color:{c}"><span style="color:{c}">{KIND_ICONS[kind]}</span> {KINDS[kind]}</span>'


def status(ok: bool) -> str:
    return '<span class="ok">✓ Atende</span>' if ok else '<span class="fail">✗ Não atende</span>'


def sym(symbol: str) -> str:
    if symbol.startswith(r"\text{") and symbol.endswith("}"):
        return html.escape(symbol[6:-1])
    return math(symbol)


def values_table(values, ctx) -> str:
    rows = "".join(
        f"<tr><td>{sym(fill(v.symbol, ctx))}</td><td>{md(table_cell(fill(v.value, ctx)))[3:-4]}</td>"
        f"<td>{badge(v.kind)}</td><td>{md(table_cell(fill(v.meaning, ctx)))[3:-4]}</td></tr>"
        for v in values)
    return ("<table class='vals'><thead><tr><th>Grandeza</th><th>Valor</th><th>Origem</th><th>Significado</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>")


def svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


# ── Figuras complementares (Matplotlib) ─────────────────────────────────────

def fig_compare_p7_13(objs) -> str:
    from dsp_av2.fir import analysis
    spec, est, fin = objs["spec"], objs["estimate"], objs["final"]
    band = spec.stopbands[0]
    w = analysis.frequency_grid(spec.edges, analysis.DENSE_GRID_POINTS)
    m = (w >= (band.lo - 0.03) * np.pi) & (w <= (band.hi + 0.03) * np.pi)
    fig, ax = plt.subplots(figsize=(7.5, 3.2), constrained_layout=True)
    for d, color, label in ((est, plotting.LIMIT if False else "#eb6834", f"estimativa M = {est.M}"),
                            (fin, plotting.SERIES, f"mínimo M = {fin.M}")):
        ax.plot(w[m] / np.pi, np.abs(analysis.frequency_response(d.h, w[m])), color=color, lw=1.5, label=label)
    ax.axhline(band.delta, color=plotting.LIMIT, ls="--", lw=1)
    ax.annotate(f"limite δ₂ = {band.delta:g}", xy=(1, band.delta), xycoords=("axes fraction", "data"),
                xytext=(-4, 3), textcoords="offset points", ha="right", fontsize=8, color=plotting.LIMIT)
    for e in (band.lo, band.hi):
        ax.axvline(e, color=plotting.LIMIT, ls=":", lw=1)
    ax.set_ylim(0, 2.2 * band.delta)
    ax.set_xlabel("ω / π")
    ax.set_ylabel("|H(e^jω)|")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    plotting._style(ax)
    return svg(fig)


def fig_poles(objs) -> str:
    p = objs["proto"].p
    r = np.max(np.abs(p))
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    th = np.linspace(np.pi / 2, 3 * np.pi / 2, 200)
    ax.plot(r * np.cos(th) / 1e3, r * np.sin(th) / 1e3, ls=":", color=plotting.LIMIT, lw=1)
    for q in p:
        ax.plot([0, q.real / 1e3], [0, q.imag / 1e3], color="#dddad2", lw=0.8)
        ang = 180 - abs(np.degrees(np.angle(q)))
        if q.imag >= 0:
            ax.annotate(f"{ang:.0f}°", (q.real / 1e3, q.imag / 1e3), xytext=(-18, 4), textcoords="offset points",
                        fontsize=7, color=plotting.LIMIT)
    ax.plot(p.real / 1e3, p.imag / 1e3, "x", color=plotting.SERIES, ms=9, mew=2)
    ax.axhline(0, color=plotting.LIMIT, lw=0.8)
    ax.axvline(0, color=plotting.LIMIT, lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15 * r / 1e3, 0.25 * r / 1e3)
    ax.set_xlabel("σ = Re(s)  (×10³ rad/s)")
    ax.set_ylabel("Im(s)  (×10³ rad/s)")
    plotting._style(ax)
    return svg(fig)


def figures_for(ex_id: str, mod, objs) -> dict[str, str]:
    """Figuras do relatório: a figura-resumo do exercício e as complementares."""
    if ex_id == "p7_13":
        overview = svg(mod_ex(ex_id).plot(objs["final"]))
        return {"impulse": overview, "amplitude": None, "db": None, "compare": fig_compare_p7_13(objs)}
    if ex_id == "p8_13":
        overview = svg(mod_ex(ex_id).plot(objs["proto"]))
        return {"poles": fig_poles(objs), "magnitude": overview, "db": None, "group_delay": None, "impulse": None}
    raise KeyError(ex_id)


def mod_ex(ex_id: str):
    import importlib
    return importlib.import_module(f"exercises.{ex_id}")


# ── Montagem do documento ───────────────────────────────────────────────────

SECTION_TITLES = ("Enunciado original", "O que está sendo pedido", "Dados e especificações",
                  "Fundamentação teórica", "Equações e variáveis", "Resolução passo a passo",
                  "Implementação computacional", "Gráficos e interpretação",
                  "Verificação das especificações", "Conclusão técnica")


def exercise_html(ex_id: str, chap_no: int, toc: list) -> str:
    mod = registry.load(ex_id)
    C, objs, ctx = mod.CONTENT, mod.objects(), mod.context()
    spec = objs["spec"]
    rep = objs.get("final_report") or objs.get("report")
    title = f"{chap_no}. Exercício {C.number}: {C.title}"
    toc.append((1, title))
    h = [f'<section class="chapter ch{C.chapter}"><div class="chap-tag">Capítulo {C.chapter} do livro · '
         f'{"FIR" if C.chapter == 7 else "IIR"}</div><h1 id="t{len(toc)}">{html.escape(title)}</h1>',
         f'<p class="tagline">{html.escape(C.tagline)}</p>']

    hl = "".join(f'<div class="hl"><div class="hl-sym">{sym(fill(v.symbol, ctx))}</div>'
                 f'<div class="hl-val">{md(fill(v.value, ctx))[3:-4]}</div>'
                 f'<div class="hl-lab">{md(fill(v.meaning, ctx))[3:-4]}</div></div>' for v in C.highlights)
    h.append(f'<p class="official">Resultado oficial: {status(rep.passed)}</p><div class="hl-grid">{hl}</div>')

    def sec(i):
        t = f"{chap_no}.{i} {SECTION_TITLES[i - 1]}"
        toc.append((2, t))
        return f'<h2 id="t{len(toc)}">{t}</h2>'

    h.append(sec(1) + f'<blockquote class="statement">{html.escape(spec.statement)}</blockquote>'
             f'<p class="src">Fonte: {html.escape(spec.source)}. Transcrição do DTA; número de página não conferido com o livro físico.</p>')
    h.append(sec(2) + md(fill(C.asked, ctx)))
    h.append(sec(3) + values_table(C.data_rows, ctx) + md(fill(C.data_note, ctx)))
    h.append(sec(4) + md(fill(C.theory, ctx)))
    for d in C.deepening:
        h.append(f'<div class="deep"><div class="deep-t">Aprofundamento: {md(d.title)[3:-4]}</div>{md(fill(d.body, ctx))}</div>')
    eqs = []
    for k, e in enumerate(C.equations, 1):
        var = "".join(f"<tr><td>{math(v.symbol)}</td><td>{md(table_cell(v.meaning))[3:-4]}</td></tr>" for v in e.variables)
        eqs.append(f'<div class="eqbox"><div class="eqname">({k}) {html.escape(e.name)}'
                   f'{" · <span class=ref>" + html.escape(e.ref) + "</span>" if e.ref else ""}</div>'
                   f'<div class="eq">{math(e.latex, True)}</div>{md(e.explanation)}'
                   + (f"<table class='vars'><tr><th>Símbolo</th><th>Significado</th></tr>{var}</table>" if var else "")
                   + "</div>")
    h.append(sec(5) + "".join(eqs))
    steps = []
    for k, s in enumerate(C.steps, 1):
        steps.append(f'<div class="step"><div class="step-h"><span>PASSO {k}</span> {md(fill(s.title, ctx))[3:-4]}</div>'
                     f'{md(fill(s.reasoning, ctx))}'
                     + (f'<div class="eq">{math(fill(s.equation, ctx), True)}</div>' if s.equation else "")
                     + (values_table(s.values, ctx) if s.values else "")
                     + (f'<div class="note">{md(fill(s.note, ctx))}</div>' if s.note else "") + "</div>")
    h.append(sec(6) + "".join(steps))
    roles = "".join(f"<tr><td><code>{r.function}</code></td><td>{html.escape(r.role)}</td></tr>" for r in C.code_roles)
    h.append(sec(7) + md(fill(C.implementation, ctx))
             + f"<table class='vars'><tr><th>Função</th><th>O que faz</th></tr>{roles}</table>")
    figs = figures_for(ex_id, mod, objs)
    parts = [sec(8)]
    nfig = 0
    for f in C.figures:
        img = figs.get(f.key)
        if img:
            nfig += 1
            cap = "Figura-resumo do exercício (Matplotlib)" if f.key in ("impulse", "magnitude") else fill(f.title, ctx)
            parts.append(f'<figure>{img}<figcaption>Figura {chap_no}.{nfig}: {md(cap)[3:-4]}</figcaption></figure>')
        parts.append(f'<h3>{md(fill(f.title, ctx))[3:-4]}</h3>{md(fill(f.reading, ctx))}')
    h.append("".join(parts))
    rows = "".join(f"<tr><td>{html.escape(c.name)}</td><td>{c.measured:.5f}</td>"
                   f"<td>{'≤' if c.relation == '<=' else '≥'} {c.limit:.5g}</td>"
                   f"<td>{round(c.margin, 9) + 0.0:+.5f}</td><td>{status(c.passed)}</td></tr>" for c in rep.checks)
    h.append(sec(9) + md(fill(C.verification, ctx))
             + f"<table class='vals'><tr><th>Critério</th><th>Medido</th><th>Limite</th><th>Folga</th><th>Situação</th></tr>{rows}</table>")
    concl = md(fill(C.conclusion, ctx))
    for eid in C.errata:
        e = errata_items()[eid]
        concl += (f'<div class="errata"><b>Correção ao documento de referência ({e.id}).</b> '
                  f'O DTA afirma {md(e.claim)[3:-4]}. <b>Correção:</b> {md(e.correction)[3:-4]}</div>')
    h.append(sec(10) + concl + "</section>")
    return "\n".join(h)


def document(ex_ids: list[str], toc_pages: dict[str, int] | None = None) -> str:
    toc: list[tuple[int, str]] = []
    P = common.PROJECT
    cover = f"""
<section class="cover">
  <div class="cv-top">{P['course']} · {P['discipline']}</div>
  <div class="cv-title">{P['app_name']}</div>
  <div class="cv-sub">{P['subtitle']}</div>
  <div class="cv-scope">Prévia do relatório: exercícios piloto {', '.join(registry.BY_ID[i].number for i in ex_ids)}</div>
  <table class="cv-id">
    <tr><td>Autor(es)</td><td>{P['authors']}</td></tr>
    <tr><td>Instituição</td><td>{P['institution']}</td></tr>
    <tr><td>Professor(a)</td><td>{P['professor']}</td></tr>
    <tr><td>Período</td><td>{P['term']}</td></tr>
    <tr><td>Livro-texto</td><td>{P['book']}</td></tr>
  </table>
  <div class="cv-date">Gerado em {date.today().strftime('%d/%m/%Y')} a partir do código e do conteúdo do projeto</div>
</section>"""
    body = []
    toc.append((1, "1. Introdução ao projeto de filtros digitais"))
    body.append(f'<section class="chapter"><h1 id="t{len(toc)}">1. Introdução ao projeto de filtros digitais</h1>'
                + md(common.PRESENTATION) + md(common.INTRODUCTION.replace("## ", "### ")) + "</section>")
    toc.append((1, "2. Metodologia"))
    legend = "".join(f"<li>{badge(k)}</li>" for k in KINDS)
    body.append(f'<section class="chapter"><h1 id="t{len(toc)}">2. Metodologia</h1>' + md(common.METHOD_SUMMARY)
                + f"<p>Cada número das resoluções traz um rótulo de origem:</p><ul class='legend'>{legend}</ul>"
                + md(common.DTA_NOTE) + "</section>")
    for n, ex_id in enumerate(ex_ids, start=3):
        body.append(exercise_html(ex_id, n, toc))
    toc.append((1, f"{len(ex_ids) + 3}. Referências"))
    refs = "".join(f"<li>{md(r)[3:-4]} <span class='refnote'>{n}</span></li>" for r, n in common.REFERENCES)
    body.append(f'<section class="chapter"><h1 id="t{len(toc)}">{len(ex_ids) + 3}. Referências</h1>'
                f"<ol class='refs'>{refs}</ol>{md(common.DTA_NOTE)}</section>")

    toc_rows = "".join(
        f'<div class="toc-l{lvl}"><a href="#t{i}">{html.escape(t)}</a><span class="dots"></span>'
        f'<span class="pg">{(toc_pages or {}).get(t, "")}</span></div>'
        for i, (lvl, t) in enumerate(toc, 1))
    toc_html = f'<section class="toc"><h1 class="toc-h">Sumário</h1>{toc_rows}</section>'
    return (f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>{P['app_name']} — Relatório</title>"
            f"<style>{CSS}</style></head><body>{cover}{toc_html}{''.join(body)}</body></html>"), toc


def print_pdf(html_path: Path, pdf_path: Path) -> None:
    chromium = shutil.which("chromium") or shutil.which("google-chrome") or shutil.which("chromium-browser")
    if not chromium:
        raise RuntimeError("Chromium não encontrado: necessário para gerar o PDF")
    subprocess.run([chromium, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                   check=True, capture_output=True, timeout=180)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def locate(pdf_path: Path, titles: list[str], skip_pages: int = 2) -> dict[str, int]:
    """Página (1-based) da primeira ocorrência de cada título, após capa e sumário."""
    pages = [_norm(p.extract_text() or "") for p in PdfReader(str(pdf_path)).pages]
    found, start = {}, skip_pages
    for t in titles:
        key = _norm(t)[:40]
        for i in range(start, len(pages)):
            if key in pages[i]:
                found[t] = i + 1
                start = i
                break
    return found


def build(out: Path = OUT_DEFAULT, ex_ids: list[str] | None = None) -> Path:
    ex_ids = ex_ids or [e.id for e in registry.EXERCISES if registry.has_content(e.id)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        html1, toc = document(ex_ids)
        (tmp / "r.html").write_text(html1, encoding="utf-8")
        print_pdf(tmp / "r.html", tmp / "pass1.pdf")
        n_toc_pages = -(-len(toc) // 36)  # teto: ~36 linhas de sumário por página
        pages = locate(tmp / "pass1.pdf", [t for _, t in toc], skip_pages=1 + n_toc_pages)
        html2, _ = document(ex_ids, pages)
        out.with_suffix(".html").write_text(html2, encoding="utf-8")
        print_pdf(out.with_suffix(".html"), out)
    missing = [t for _, t in toc if t not in pages]
    if missing:
        print("aviso: títulos não localizados no PDF:", missing)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    print("gerado:", build(args.out))


if __name__ == "__main__":
    main()
