"""1. Apresentação do projeto."""

import streamlit as st
from streamlit.errors import StreamlitPageNotFoundError

from app.components.style import FIR_ACCENT, IIR_ACCENT, inject_css, kind_legend
from content import registry
from content.common import METHOD_SUMMARY, PRESENTATION, PROJECT

inject_css()
st.markdown(f"<div class='ex-number'>{PROJECT['discipline'].upper()} · {PROJECT['course'].upper()}</div>",
            unsafe_allow_html=True)
st.title(PROJECT["app_name"])
st.markdown(f"<div class='ex-tagline' style='font-size:1.1rem'>{PROJECT['subtitle']}</div>", unsafe_allow_html=True)
st.write("")
st.markdown(PRESENTATION)

st.header("Os 12 exercícios", anchor="exercicios")
c7, c8 = st.columns(2)
for col, chapter, color, label in ((c7, 7, FIR_ACCENT, "Capítulo 7 — FIR Filter Design"),
                                   (c8, 8, IIR_ACCENT, "Capítulo 8 — IIR Filter Design")):
    with col:
        st.markdown(f"<div style='border-top:4px solid {color};padding-top:.4rem'><b>{label}</b></div>",
                    unsafe_allow_html=True)
        for e in registry.EXERCISES:
            if e.chapter != chapter:
                continue
            label = f"{e.number} · {e.short} · {e.method}"
            if registry.has_content(e.id):
                try:
                    st.page_link(f"pages/{e.id}.py", label=label, icon=":material/arrow_forward:")
                except StreamlitPageNotFoundError:  # página executada fora da navegação (testes)
                    st.markdown(f"→ [{label}](/{e.id})")
            else:
                st.markdown(f"<span style='color:#8a8880'>{label} <i>(em breve)</i></span>", unsafe_allow_html=True)

st.header("Análise", anchor="analise")
for page, label in (("fir_vs_iir", "Comparação FIR × IIR"), ("validacao", "Validação dos resultados"),
                    ("referencias", "Referências e metodologia")):
    try:
        st.page_link(f"pages/{page}.py", label=label, icon=":material/arrow_forward:")
    except StreamlitPageNotFoundError:  # página executada fora da navegação (testes)
        st.markdown(f"→ [{label}](/{page})")

st.header("Metodologia em uma página", anchor="metodologia")
st.markdown(METHOD_SUMMARY)
st.markdown("**Origem dos valores.** Cada número das resoluções traz um rótulo:")
st.markdown(kind_legend(), unsafe_allow_html=True)

st.header("Identificação", anchor="identificacao")
st.markdown("\n".join([
    "| | |", "|---|---|",
    f"| Livro-texto | {PROJECT['book']} |",
    f"| Exercícios | {PROJECT['exercises']} |",
    f"| Curso / disciplina | {PROJECT['course']} — {PROJECT['discipline']} |",
    f"| Autor(es) | {PROJECT['authors']} |",
    f"| Instituição | {PROJECT['institution']} |",
    f"| Professor(a) | {PROJECT['professor']} |",
    f"| Período | {PROJECT['term']} |",
]))
st.caption("Campos marcados ⟨a preencher⟩ dependem de informações que ainda não foram fornecidas.")
