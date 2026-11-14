"""Identidade visual: CSS global, rótulos de origem dos valores e indicadores.

Cores definidas uma única vez aqui. Cada rótulo combina cor, ícone e texto:
a informação nunca depende só da cor.
"""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from content.model import KINDS

FIR_ACCENT = "#1f5fae"
IIR_ACCENT = "#0f7a5a"
INK = "#1c1c1a"
MUTED = "#5a5953"
LINE = "#e2e0da"
PAPER = "#fbfaf7"

KIND_STYLE = {   # (ícone, cor da borda, fundo)
    "dado": ("■", "#2a78d6", "#eef4fc"),
    "calculado": ("∑", "#4a3aa7", "#f1effb"),
    "escolhido": ("◆", "#b07800", "#fdf6e6"),
    "resultado": ("●", "#0f7a5a", "#e9f6f1"),
    "criterio": ("✓", "#8a3b12", "#fbefe8"),
}

_CSS = (Path(__file__).with_name("theme.css")).read_text(encoding="utf-8")


def inject_css() -> None:
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def kind_badge(kind: str) -> str:
    icon, color, bg = KIND_STYLE[kind]
    return (f'<span class="kind" style="border-color:{color};background:{bg}">'
            f'<span class="kind-icon" style="color:{color}">{icon}</span>{html.escape(KINDS[kind])}</span>')


def status_badge(passed: bool) -> str:
    return ('<span class="status ok">✓ Atende</span>' if passed
            else '<span class="status fail">✗ Não atende</span>')


def chapter_badge(chapter: int) -> str:
    if chapter == 7:
        return f'<span class="chapter" style="background:{FIR_ACCENT}">Capítulo 7 · FIR</span>'
    return f'<span class="chapter" style="background:{IIR_ACCENT}">Capítulo 8 · IIR</span>'


def accent(chapter: int) -> str:
    return FIR_ACCENT if chapter == 7 else IIR_ACCENT


def kind_legend() -> str:
    return '<div class="legend">' + "".join(kind_badge(k) for k in KINDS) + "</div>"
