"""Páginas de análise (Etapa 5): renderizam os blocos de ``content.analysis``."""

from __future__ import annotations

import streamlit as st

from . import figures
from .style import inject_css


@st.cache_resource(show_spinner="Calculando…")
def _blocks(name: str):
    from content import analysis
    return {"comparacao": analysis.comparison_blocks, "validacao": analysis.validation_blocks,
            "metodologia": analysis.methodology_blocks}[name]()


def render(name: str, label: str, title: str, lead: str) -> None:
    inject_css()
    st.markdown(f"<div class='ex-number'>{label}</div>", unsafe_allow_html=True)
    st.title(title)
    st.markdown(f"<div class='ex-tagline'>{lead}</div>", unsafe_allow_html=True)
    for i, b in enumerate(_blocks(name)):
        if b.kind == "md":
            st.markdown(b.text, unsafe_allow_html=True)
        else:
            st.subheader(b.title)
            st.plotly_chart(figures.from_data(b.key, b.data), use_container_width=True, key=f"{name}_{i}",
                            config={"displaylogo": False})
            st.caption(b.text)
