"""Explore — P7.13: efeito de M e de β sobre o atendimento das tolerâncias."""

from __future__ import annotations

import numpy as np
import streamlit as st

from dsp_av2.fir import analysis, windows
from dsp_av2.fir.design import ideal_response
from dsp_av2.specs import P7_13 as SPEC

from ..components import figures
from ..components.style import status_badge


def render(objs: dict, ctx: dict) -> None:
    est, final = objs["estimate"], objs["final"]
    c1, c2 = st.columns(2)
    M = c1.select_slider("Comprimento M (só ímpares: rejeita-faixa exige Tipo 1)",
                         options=list(range(45, 73, 2)), value=est.M, key="x713_M")
    beta_mode = c2.radio("Parâmetro β", ["Fórmula de Kaiser", "Valor livre"], horizontal=True, key="x713_bmode")
    beta = est.params["beta"]
    if beta_mode == "Valor livre":
        beta = c2.slider("β", 2.0, 8.0, float(round(beta, 2)), 0.05, key="x713_beta")
    else:
        c2.markdown(f"β = {beta:.4f} (A_s = {ctx['As']:.2f} dB)")

    h = ideal_response(SPEC, est.cutoffs, M) * windows.kaiser(M, beta)
    w = analysis.frequency_grid(SPEC.edges, analysis.DENSE_GRID_POINTS)
    devs = analysis.band_deviations(h, SPEC.bands, w)
    rows = ["| Faixa | Desvio máximo | Tolerância | Situação |", "|---|---|---|---|"]
    for b, d in zip(SPEC.bands, devs):
        rows.append(f"| [{b.lo:g}π, {b.hi:g}π] | {d:.5f} | {b.delta:g} | {status_badge(d <= b.delta)} |")
    st.markdown("\n".join(rows), unsafe_allow_html=True)
    st.plotly_chart(figures.fir_compare_stopband(w, {f"explorado M = {M}, β = {beta:.3f}": h,
                                                     f"oficial M = {final.M}": final.h},
                                                 SPEC.stopbands[0], SPEC.stopbands[0].delta),
                    use_container_width=True, key="x713_fig")
    st.caption(f"Experimente: com β da fórmula, M = {ctx['M_est']} falha e M = {ctx['M']} é o primeiro que atende. "
               "Aumentar β reduz os lóbulos, mas alarga a transição.")
