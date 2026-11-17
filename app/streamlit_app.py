"""DSP Filter Lab — ponto de entrada.

Execução (na raiz do projeto):  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

from app.components.exercise import render_pending  # noqa: E402
from content import registry  # noqa: E402

st.set_page_config(page_title="DSP Filter Lab", page_icon="〰", layout="centered",
                   initial_sidebar_state="auto")

PAGES_DIR = Path(__file__).parent / "pages"


def _pending_page(ex_id: str):
    def page():
        render_pending(ex_id)
    page.__name__ = f"pending_{ex_id}"
    return page


def _exercise_page(e: registry.Entry) -> st.Page:
    title = f"{e.number} · {e.short} · {e.method}"
    file = PAGES_DIR / f"{e.id}.py"
    if file.exists():
        return st.Page(str(file), title=title, url_path=e.id)
    return st.Page(_pending_page(e.id), title=f"{title}  (em breve)", url_path=e.id)


nav = st.navigation({
    "Início": [
        st.Page(str(PAGES_DIR / "apresentacao.py"), title="Apresentação", icon=":material/home:", default=True),
        st.Page(str(PAGES_DIR / "introducao.py"), title="Introdução aos filtros digitais", icon=":material/menu_book:"),
    ],
    "Capítulo 7 — FIR": [_exercise_page(e) for e in registry.EXERCISES if e.chapter == 7],
    "Capítulo 8 — IIR": [_exercise_page(e) for e in registry.EXERCISES if e.chapter == 8],
    "Análise": [
        st.Page(str(PAGES_DIR / "fir_vs_iir.py"), title="Comparação FIR × IIR", icon=":material/compare_arrows:",
                url_path="fir_vs_iir"),
        st.Page(str(PAGES_DIR / "validacao.py"), title="Validação dos resultados", icon=":material/fact_check:",
                url_path="validacao"),
        st.Page(str(PAGES_DIR / "referencias.py"), title="Referências e metodologia", icon=":material/library_books:",
                url_path="referencias"),
    ],
})
nav.run()
