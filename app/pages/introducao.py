"""2. Introdução ao projeto de filtros digitais."""

import streamlit as st

from app.components.style import inject_css
from content.common import INTRODUCTION

inject_css()
st.markdown("<div class='ex-number'>INTRODUÇÃO</div>", unsafe_allow_html=True)
st.title("Projeto de filtros digitais")
st.markdown(INTRODUCTION)
