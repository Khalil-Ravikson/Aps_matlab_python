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


# Formato das figuras embutidas: "svg" (PDF, vetorial) ou "png" (DOCX, que não aceita SVG em HTML).
FIGURE_FORMAT = "svg"


def svg(fig) -> str:
    """Figura Matplotlib embutida no HTML (SVG vetorial ou PNG em data URI, conforme FIGURE_FORMAT)."""
    if FIGURE_FORMAT == "png":
        import base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        width_in = fig.get_size_inches()[0]
        return (f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" '
                f'style="width:{min(width_in, 6.3):.2f}in" alt="figura">')
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


# ── Figuras complementares (Matplotlib), guiadas pelos dados de content.objects() ──

STANDARD = {"impulse", "amplitude", "db", "passband", "magnitude", "phase", "group_delay"}


def _ax(figsize=(7.5, 3.2)):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return fig, ax


def fig_compare(objs) -> str:
    from dsp_av2.fir import analysis
    c, spec = objs["compare"], objs["spec"]
    band = c["band"]
    w = analysis.frequency_grid(spec.edges, analysis.DENSE_GRID_POINTS)
    m = (w >= (band.lo - 0.03) * np.pi) & (w <= (band.hi + 0.03) * np.pi)
    fig, ax = _ax()
    for (label, h), color in zip(c["designs"].items(), ("#eb6834", plotting.SERIES)):
        ax.plot(w[m] / np.pi, np.abs(analysis.frequency_response(h, w[m])), color=color, lw=1.5, label=label)
    ax.axhline(c["limit"], color=plotting.LIMIT, ls="--", lw=1)
    for e in (band.lo, band.hi):
        ax.axvline(e, color=plotting.LIMIT, ls=":", lw=1)
    ax.set_ylim(0, 2.2 * c["limit"])
    ax.set_xlabel("ω / π")
    ax.set_ylabel("|H(e^jω)|")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    plotting._style(ax)
    return svg(fig)


def fig_scan(d) -> str:
    x, y = np.asarray(d["x"]), np.asarray(d["y"], float)
    ok = (y >= d["limit"]) if d["better"] == "above" else (y <= d["limit"])
    fig, ax = _ax((7.5, 3.0))
    ax.plot(x, y, color="#cfccc4", lw=1.2)
    if ok.any():
        ax.plot(x[ok], y[ok], "o", color="#0f7a5a", ms=6, label="atende")
    if (~ok).any():
        ax.plot(x[~ok], y[~ok], "x", color="#b3261e", ms=7, mew=2, label="não atende")
    ax.axhline(d["limit"], color=plotting.LIMIT, ls="--", lw=1)
    if d.get("highlight") is not None:
        ax.axvline(d["highlight"], color=plotting.SERIES, ls=":", lw=1.2)
    ax.set_xlabel(d["xlabel"])
    ax.set_ylabel(d["ylabel"])
    ax.legend(frameon=False, fontsize=8)
    plotting._style(ax)
    return svg(fig)


def fig_overlay(d) -> str:
    from dsp_av2.fir import analysis
    w = np.linspace(d["xlim"][0] * np.pi, d["xlim"][1] * np.pi, 6001)
    fig, ax = _ax()
    for (label, h), color in zip(d["curves"].items(), (plotting.SERIES, "#eb6834", "#1baf7a")):
        H = analysis.frequency_response(h, w)
        with np.errstate(divide="ignore"):
            db = analysis.magnitude_db(H) if d.get("normalized") else 20 * np.log10(np.abs(H))
        ax.plot(w / np.pi, db, color=color, lw=1.4, label=label)
    if d.get("limit_db") is not None:
        ax.axhline(d["limit_db"], color=plotting.LIMIT, ls="--", lw=1)
    for e in d.get("edges", []):
        ax.axvline(e, color=plotting.LIMIT, ls=":", lw=1)
    ax.set_ylim(*d["ylim"])
    ax.set_xlim(*d["xlim"])
    ax.set_xlabel("ω / π")
    ax.set_ylabel("|H| (dB)")
    if len(d["curves"]) > 1:
        ax.legend(frameon=False, fontsize=8)
    plotting._style(ax)
    return svg(fig)


def fig_bars(d) -> str:
    cats = d["categories"]
    x = np.arange(len(cats))
    n = len(d["series"])
    width = 0.8 / n
    fig, ax = _ax((7.0, 3.0))
    for k, ((label, vals), color) in enumerate(zip(d["series"].items(), ("#9fbfe6", plotting.SERIES, "#eb6834"))):
        bars = ax.bar(x + (k - (n - 1) / 2) * width, vals, width, label=label, color=color)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x, cats)
    ax.set_ylabel(d["ylabel"])
    ax.legend(frameon=False, fontsize=8)
    plotting._style(ax)
    return svg(fig)


def fig_lines(d) -> str:
    fig, ax = _ax((7.5, 3.4))
    for (label, y), color in zip(d["curves"].items(), (plotting.SERIES, "#eb6834", "#1baf7a", "#4a3aa7")):
        ax.plot(d["x"], y, color=color, lw=1.4, label=label)
    for y, _ in d.get("hlines", []):
        ax.axhline(y, color=plotting.LIMIT, ls="--", lw=1)
    for e in d.get("vlines", []):
        ax.axvline(e, color=plotting.LIMIT, ls=":", lw=1)
    if d.get("ylim"):
        ax.set_ylim(*d["ylim"])
    ax.set_xlabel(d["xlabel"])
    ax.set_ylabel(d["ylabel"])
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    plotting._style(ax)
    return svg(fig)


def blocks_html(blocks, chap_no: int, fig_counter: list) -> str:
    """Seções de análise: Markdown e figuras (mesmos blocos da aplicação)."""
    out = []
    for b in blocks:
        if b.kind == "md":
            out.append(md(b.text.replace("\n## ", "\n### ")))
        else:
            fig_counter[0] += 1
            img = fig_bars(b.data) if b.key.startswith("bars") else fig_lines(b.data)
            out.append(f'<figure>{img}<figcaption>Figura {chap_no}.{fig_counter[0]}: {html.escape(b.title)}. '
                       f'{md(b.text)[3:-4]}</figcaption></figure>')
    return "".join(out)


def fig_poles(objs) -> str:
    p = objs["proto"].p
    r = np.max(np.abs(p))
    curve = objs.get("pole_curve") or ("circle", r)
    ax_r, ax_i = (curve[1], curve[1]) if curve[0] == "circle" else (curve[1], curve[2])
    sc = 1e3 if r > 1e3 else 1.0
    unit = "×10³ rad/s" if sc > 1 else "rad/s"
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    th = np.linspace(np.pi / 2, 3 * np.pi / 2, 200)
    ax.plot(ax_r * np.cos(th) / sc, ax_i * np.sin(th) / sc, ls=":", color=plotting.LIMIT, lw=1)
    for q in p:
        ax.plot([0, q.real / sc], [0, q.imag / sc], color="#dddad2", lw=0.8)
    ax.plot(p.real / sc, p.imag / sc, "x", color=plotting.SERIES, ms=9, mew=2)
    ax.axhline(0, color=plotting.LIMIT, lw=0.8)
    ax.axvline(0, color=plotting.LIMIT, lw=0.8)
    ax.set_aspect("equal")
    xr = max(ax_r, np.max(-p.real))
    ax.set_xlim(-1.25 * xr / sc, 0.35 * xr / sc)
    ax.set_ylim(-1.1 * r / sc, 1.1 * r / sc)
    ax.set_xlabel(f"σ = Re(s)  ({unit})")
    ax.set_ylabel(f"Im(s)  ({unit})")
    plotting._style(ax)
    return svg(fig)


def figures_for(ex_id: str, objs) -> dict[str, str]:
    """Figura de cada chave: a figura-resumo do exercício entra uma vez (na 1ª chave padrão)."""
    mod = registry.load(ex_id)
    out, overview_done = {}, False
    for f in mod.CONTENT.figures:
        k = f.key
        if k in STANDARD:
            if not overview_done:
                out[k] = svg(mod_ex(ex_id).plot(objs.get("final") or objs["proto"]))
                overview_done = True
            else:
                out[k] = None
        elif k == "poles":
            out[k] = fig_poles(objs)
        elif k == "compare":
            out[k] = fig_compare(objs)
        elif k.startswith("scan"):
            out[k] = fig_scan(objs[k])
        elif k.startswith("overlay"):
            out[k] = fig_overlay(objs[k])
        elif k.startswith("bars"):
            out[k] = fig_bars(objs[k])
        else:
            raise KeyError(f"figura sem versão Matplotlib: {ex_id}/{k}")
    return out


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
    figs = figures_for(ex_id, objs)
    parts = [sec(8)]
    nfig = 0
    for f in C.figures:
        img = figs.get(f.key)
        if img:
            nfig += 1
            cap = ("Figura-resumo do exercício: gráficos pedidos pelo enunciado" if f.key in STANDARD
                   else fill(f.title, ctx))
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


TOC_END = "FIMDOSUMARIO"


def document(ex_ids: list[str]) -> tuple[str, list]:
    """HTML completo; os números de página do sumário ficam como marcadores \\x01i\\x01."""
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
    from content import analysis
    meth = analysis.methodology_blocks()
    toc.append((1, "2. Metodologia"))
    legend = "".join(f"<li>{badge(k)}</li>" for k in KINDS)
    body.append(f'<section class="chapter"><h1 id="t{len(toc)}">2. Metodologia</h1>' + blocks_html(meth[:1], 2, [0])
                + f"<p>Cada número das resoluções traz um rótulo de origem:</p><ul class='legend'>{legend}</ul>"
                + blocks_html(meth[1:3], 2, [0]) + "</section>")
    for n, ex_id in enumerate(ex_ids, start=3):
        body.append(exercise_html(ex_id, n, toc))
    n = len(ex_ids) + 3
    if len(ex_ids) == len(registry.EXERCISES):   # seções de análise exigem os 12 exercícios
        for title, blocks in (("Comparação FIR × IIR", analysis.comparison_blocks()),
                              ("Validação dos resultados", analysis.validation_blocks())):
            toc.append((1, f"{n}. {title}"))
            body.append(f'<section class="chapter"><h1 id="t{len(toc)}">{n}. {title}</h1>'
                        + blocks_html(blocks, n, [0]) + "</section>")
            n += 1
    toc.append((1, f"{n}. Referências e limitações"))
    refs = "".join(f"<li>{md(r)[3:-4]} <span class='refnote'>{note}</span></li>" for r, note in common.REFERENCES)
    lim = meth[3].text.split("## Limitações", 1)[1]
    body.append(f'<section class="chapter"><h1 id="t{len(toc)}">{n}. Referências e limitações</h1>'
                f"<ol class='refs'>{refs}</ol>{md(common.DTA_NOTE)}<h2>Limitações</h2>{md(lim)}</section>")

    toc_rows = "".join(
        f'<div class="toc-l{lvl}"><a href="#t{i}">{html.escape(t)}</a><span class="dots"></span>'
        f'<span class="pg">\x01{i}\x01</span></div>'
        for i, (lvl, t) in enumerate(toc, 1))
    toc_html = (f'<section class="toc"><h1 class="toc-h">Sumário</h1>{toc_rows}'
                f'<div class="toc-end">{TOC_END}</div></section>')
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


def locate(pdf_path: Path, titles: list[str]) -> dict[str, int]:
    """Página (1-based) da primeira ocorrência de cada título, depois do fim do sumário."""
    pages = [_norm(p.extract_text() or "") for p in PdfReader(str(pdf_path)).pages]
    start = next(i for i, p in enumerate(pages) if TOC_END in p) + 1
    found = {}
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
    html_doc, toc = document(ex_ids)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "r.html").write_text(re.sub("\x01\\d+\x01", "000", html_doc), encoding="utf-8")
        print_pdf(tmp / "r.html", tmp / "pass1.pdf")
        pages = locate(tmp / "pass1.pdf", [t for _, t in toc])
    final = re.sub("\x01(\\d+)\x01", lambda m: str(pages.get(toc[int(m.group(1)) - 1][1], "")), html_doc)
    out.with_suffix(".html").write_text(final, encoding="utf-8")
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
