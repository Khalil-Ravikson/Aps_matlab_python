"""Página de exercício: as 10 seções padronizadas, a partir de ``content/<id>.py``.

Este módulo só organiza a apresentação. Textos vêm de ``content``, números do
núcleo (``context()``/``objects()``), verificações de ``dsp_av2.validation``
e figuras de ``figures``.
"""

from __future__ import annotations

import importlib
import inspect

import streamlit as st

from content import registry
from content.errata import items as errata_items
from content.model import KINDS, fill, table_cell
from dsp_av2.specs import AnalogLowpassSpec

from . import figures
from .style import accent, chapter_badge, inject_css, kind_badge, kind_legend, status_badge

SECTIONS = (
    ("s1", "Enunciado original"), ("s2", "O que está sendo pedido"), ("s3", "Dados e especificações"),
    ("s4", "Fundamentação teórica"), ("s5", "Equações e variáveis"), ("s6", "Resolução passo a passo"),
    ("s7", "Implementação computacional"), ("s8", "Gráficos e interpretação"),
    ("s9", "Verificação das especificações"), ("s10", "Conclusão técnica"),
)


@st.cache_resource(show_spinner="Calculando o projeto…")
def _load(ex_id: str):
    mod = registry.load(ex_id)
    return mod, mod.objects(), mod.context()


def _md(text: str, ctx: dict) -> None:
    st.markdown(fill(text, ctx), unsafe_allow_html=True)


def _values_table(values, ctx: dict, header=("Grandeza", "Valor", "Origem", "Significado")) -> None:
    rows = [f"| {' | '.join(header)} |", "|---|---|---|---|"]
    for v in values:
        val = fill(v.value, ctx)
        rows.append(f"| {_sym(fill(v.symbol, ctx))} | {table_cell(val)} | {kind_badge(v.kind)} | "
                    f"{table_cell(fill(v.meaning, ctx))} |")
    st.markdown("\n".join(rows), unsafe_allow_html=True)


def _sym(symbol: str) -> str:
    """Célula de símbolo: rótulos textuais saem como texto; símbolos, como fórmula."""
    if symbol.startswith(r"\text{") and symbol.endswith("}"):
        return table_cell(symbol[6:-1])
    return table_cell(f"${symbol}$")


def _section(i: int) -> None:
    key, title = SECTIONS[i - 1]
    st.header(f"{i}. {title}", anchor=key)


def _report(objs: dict):
    return objs.get("final_report") or objs.get("report")


def render_exercise(ex_id: str) -> None:
    inject_css()
    mod, objs, ctx = _load(ex_id)
    C = mod.CONTENT
    spec = objs["spec"]

    # ── Cabeçalho ──────────────────────────────────────────────────────────
    st.markdown(f'<div class="ex-header">{chapter_badge(C.chapter)}'
                f'<span class="ex-number">EXERCÍCIO {C.number}</span></div>', unsafe_allow_html=True)
    st.title(C.title)
    st.markdown(f'<div class="ex-tagline">{C.tagline}</div>', unsafe_allow_html=True)

    rep = _report(objs)
    st.markdown(f"**Resultado oficial** &nbsp; {status_badge(rep.passed)} &nbsp; "
                f"<span style='color:#5a5953;font-size:.85rem'>todas as especificações verificadas na seção 9</span>",
                unsafe_allow_html=True)
    cols = st.columns(len(C.highlights))
    for col, v in zip(cols, C.highlights):
        with col.container(border=True):
            st.markdown(f"${fill(v.symbol, ctx)}$")
            st.markdown(f"<div class='hl-value'>{fill(v.value, ctx)}</div>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size:.8rem;color:#5a5953'>{fill(v.meaning, ctx)}</span>",
                        unsafe_allow_html=True)

    st.markdown('<div class="toc">' + "".join(f'<a href="#{k}">{i}. {t}</a>' for i, (k, t) in enumerate(SECTIONS, 1))
                + '<a href="#explore">Explore</a></div>', unsafe_allow_html=True)
    with st.expander("Como ler os rótulos de origem dos valores"):
        st.markdown("Cada número da resolução traz um rótulo que indica de onde ele vem:", unsafe_allow_html=True)
        st.markdown(kind_legend(), unsafe_allow_html=True)

    # 1 ────────────────────────────────────────────────────────────────────
    _section(1)
    st.markdown(f'<div class="statement">{spec.statement}</div>'
                f'<div class="statement-src">Fonte: {spec.source}. Transcrição do DTA; '
                f'número de página não conferido com o livro físico.</div>', unsafe_allow_html=True)

    # 2 ────────────────────────────────────────────────────────────────────
    _section(2)
    _md(C.asked, ctx)

    # 3 ────────────────────────────────────────────────────────────────────
    _section(3)
    _values_table(C.data_rows, ctx)
    st.caption("")
    _md(C.data_note, ctx)

    # 4 ────────────────────────────────────────────────────────────────────
    _section(4)
    _md(C.theory, ctx)
    for d in C.deepening:
        with st.expander(d.title):
            _md(d.body, ctx)

    # 5 ────────────────────────────────────────────────────────────────────
    _section(5)
    st.markdown("As equações usadas na resolução, na ordem em que aparecem. Cada uma traz o significado das variáveis.")
    for k, e in enumerate(C.equations, 1):
        with st.container(border=True):
            st.markdown(f"**({k}) {e.name}**" + (f" <span style='color:#5a5953;font-size:.8rem'>· {e.ref}</span>" if e.ref else ""),
                        unsafe_allow_html=True)
            st.latex(e.latex)
            _md(e.explanation, ctx)
            if e.variables:
                st.markdown("\n".join(["| Símbolo | Significado |", "|---|---|"]
                                      + [f"| {table_cell('$' + v.symbol + '$')} | {table_cell(v.meaning)} |"
                                         for v in e.variables]))

    # 6 ────────────────────────────────────────────────────────────────────
    _section(6)
    st.markdown("Cada passo mostra **por que** é feito, a conta com os números substituídos e a origem de cada valor.")
    for k, s in enumerate(C.steps, 1):
        with st.container(border=True):
            st.markdown(f"<span style='color:{accent(C.chapter)};font-weight:700'>PASSO {k}</span> &nbsp; "
                        f"**{fill(s.title, ctx)}**", unsafe_allow_html=True)
            _md(s.reasoning, ctx)
            if s.equation:
                st.latex(fill(s.equation, ctx))
            if s.values:
                _values_table(s.values, ctx)
            if s.note:
                st.warning(fill(s.note, ctx), icon=":material/report:")

    # 7 ────────────────────────────────────────────────────────────────────
    _section(7)
    _md(C.implementation, ctx)
    st.markdown("\n".join(["| Função | O que faz |", "|---|---|"]
                          + [f"| `{r.function}` | {table_cell(r.role)} |" for r in C.code_roles]))
    with st.expander("Ver o código-fonte (opcional)"):
        for r in C.code_roles:
            mod_name, fn = r.function.rsplit(".", 1)
            obj = getattr(importlib.import_module(mod_name), fn)
            st.markdown(f"**`{r.function}`**: {r.role}")
            st.code(inspect.getsource(obj), language="python")

    # 8 ────────────────────────────────────────────────────────────────────
    _section(8)
    for f in C.figures:
        st.subheader(fill(f.title, ctx))
        st.plotly_chart(figures.build(ex_id, f.key, objs), use_container_width=True,
                        config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 3}})
        _md(f.reading, ctx)

    # 9 ────────────────────────────────────────────────────────────────────
    _section(9)
    _md(C.verification, ctx)
    rows = ["| Critério | Medido | Limite | Folga | Situação |", "|---|---|---|---|---|"]
    for c in rep.checks:
        rel = "≤" if c.relation == "<=" else "≥"
        rows.append(f"| {table_cell(c.name)} | {c.measured:.5f} | {rel} {c.limit:.5g} | "
                    f"{round(c.margin, 9) + 0.0:+.5f} | {status_badge(c.passed)} |")
    st.markdown("\n".join(rows), unsafe_allow_html=True)
    st.markdown(f"Rótulo de todas as linhas: {kind_badge('criterio')}", unsafe_allow_html=True)

    # 10 ───────────────────────────────────────────────────────────────────
    _section(10)
    _md(C.conclusion, ctx)
    for eid in C.errata:
        e = errata_items()[eid]
        with st.container(border=True):
            st.markdown(f"<span style='color:#9b1c1c;font-weight:700'>CORREÇÃO AO DOCUMENTO DE REFERÊNCIA · {e.id}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"**O DTA afirma:** {e.claim}\n\n**Correção:** {e.correction}")
            st.caption(f"Evidência: {e.evidence} · Detalhes em docs/errata_dta.md")

    # Explore ──────────────────────────────────────────────────────────────
    st.header("Explore", anchor="explore")
    st.markdown('<div class="explore-banner"><b>Modo de exploração.</b> Altere os parâmetros e veja o efeito. '
                'Nada aqui muda o <b>resultado oficial</b> apresentado acima.</div>', unsafe_allow_html=True)
    try:
        explore = importlib.import_module(f"app.explore.{ex_id}")
    except ModuleNotFoundError:
        st.info("Exploração ainda não disponível para este exercício.")
    else:
        explore.render(objs, ctx)


def render_pending(ex_id: str) -> None:
    """Página provisória de um exercício cujo conteúdo didático ainda será escrito."""
    inject_css()
    e = registry.BY_ID[ex_id]
    st.markdown(f'<div class="ex-header">{chapter_badge(e.chapter)}'
                f'<span class="ex-number">EXERCÍCIO {e.number}</span></div>', unsafe_allow_html=True)
    st.title(f"{e.number} · {e.short} · {e.method}")
    st.info("A página didática deste exercício será desenvolvida depois da aprovação dos pilotos "
            "(P7.13 e P8.13). A resolução já existe e está validada no projeto: script em "
            f"`exercises/{ex_id}.py` e notebook em `notebooks/{ex_id}.ipynb`.")
