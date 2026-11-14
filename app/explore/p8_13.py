"""Explore — P8.13: efeito da ordem N sobre a atenuação em Fs."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dsp_av2.iir import analysis
from dsp_av2.iir.prototypes import butterworth_of_order
from dsp_av2.specs import P8_13 as SPEC

from ..components import figures
from ..components.style import status_badge


def render(objs: dict, ctx: dict) -> None:
    official = objs["proto"]
    N = st.slider("Ordem N do Butterworth (Ω_c recalculado para R_p exato em Ω_p)", 4, 14, official.N, key="x813_N")
    proto = butterworth_of_order(N, SPEC.wp, SPEC.rp_db)
    att = -analysis.magnitude_db(analysis.frequency_response(proto, np.array([SPEC.ws])))[0]
    st.markdown(f"Atenuação em {SPEC.ws / 2 / np.pi / 1e3:g} kHz: **{att:.2f} dB** (exigido ≥ {SPEC.as_db:g} dB) "
                f"&nbsp; {status_badge(att >= SPEC.as_db)}", unsafe_allow_html=True)
    w = np.linspace(0, 2 * SPEC.ws, 4001)
    fig = figures.analog_db(official, SPEC, w)
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(np.abs(analysis.frequency_response(proto, w)))
    fig.data[0].name = f"oficial N = {official.N}"
    fig.add_trace(go.Scatter(x=w / 2 / np.pi, y=db, mode="lines", name=f"explorado N = {N}",
                             line=dict(color=figures.SERIES2, width=2, dash="dash")))
    fig.update_layout(showlegend=True, margin=dict(t=60), legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig, use_container_width=True, key="x813_fig")
    st.caption(f"Cada unidade a mais em N acrescenta cerca de 6 dB por oitava de queda. Com N = {ctx['Nm1']} "
               f"a atenuação fica abaixo de {ctx['As']:g} dB; com N = {ctx['N']} passa, por pouco.")
