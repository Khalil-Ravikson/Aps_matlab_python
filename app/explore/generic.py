"""Explorações reutilizáveis. Usam só funções do núcleo e nunca alteram o resultado oficial."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dsp_av2.fir import analysis as fa
from dsp_av2.fir import windows as W
from dsp_av2.fir.design import admissible_lengths, design_fixed_window, ideal_response, requires_odd_length
from dsp_av2.iir import analysis as ia
from dsp_av2.iir.prototypes import OF_ORDER

from ..components import figures
from ..components.style import status_badge

WINDOW_NAMES = {"hann": "Hann", "hamming": "Hamming", "blackman": "Blackman"}


def _lengths(spec, lo, hi):
    return list(admissible_lengths(spec, lo, hi))


def fir_fixed(objs, prefix: str, windows=("hann", "hamming", "blackman"), span=(0.5, 1.4)) -> None:
    """Janela fixa: escolher janela e M; medir Rp e As como no projeto oficial."""
    spec, final = objs["spec"], objs["final"]
    c1, c2 = st.columns(2)
    win = c1.radio("Janela", windows, index=windows.index(final.window), format_func=WINDOW_NAMES.get,
                   horizontal=True, key=f"{prefix}_win")
    opts = _lengths(spec, int(span[0] * final.M), int(span[1] * final.M))
    note = " (só ímpares: H(π) ≠ 0 exige Tipo 1)" if requires_odd_length(spec) else " (pares e ímpares admissíveis)"
    M = c2.select_slider("Comprimento M" + note, options=opts, value=min(opts, key=lambda m: abs(m - final.M)),
                         key=f"{prefix}_M")
    d = design_fixed_window(spec, win, M=M)
    rp, as_ = fa.measure_rp_as(d.h, spec.passbands, spec.stopbands, fa.frequency_grid(spec.edges, fa.DENSE_GRID_POINTS))
    st.markdown("\n".join([
        "| Grandeza | Medido | Exigido | Situação |", "|---|---|---|---|",
        f"| Rp | {rp:.4f} dB | ≤ {spec.rp_db:g} dB | {status_badge(rp <= spec.rp_db)} |",
        f"| As | {as_:.2f} dB | ≥ {spec.as_db:g} dB | {status_badge(as_ >= spec.as_db)} |",
    ]), unsafe_allow_html=True)
    fig = figures.overlay_db({"curves": {f"explorado: {WINDOW_NAMES[win]}, M = {M}": d.h,
                                         f"oficial: {WINDOW_NAMES[final.window]}, M = {final.M}": final.h},
                              "xlim": (0, 1), "ylim": (-100, 5), "normalized": True,
                              "limit_db": -spec.as_db, "limit_label": f"−As = −{spec.as_db:g} dB",
                              "edges": list(spec.edges)})
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_fig")


def fir_kaiser(objs, prefix: str) -> None:
    """Kaiser: escolher M e β; medir os desvios de cada faixa."""
    spec, est, final = objs["spec"], objs["estimate"], objs["final"]
    c1, c2 = st.columns(2)
    opts = _lengths(spec, int(0.75 * final.M), int(1.2 * final.M))
    note = " (só ímpares)" if requires_odd_length(spec) else " (pares e ímpares)"
    M = c1.select_slider("Comprimento M" + note, options=opts, value=min(opts, key=lambda m: abs(m - est.M)),
                         key=f"{prefix}_M")
    mode = c2.radio("Parâmetro β", ["Fórmula de Kaiser", "Valor livre"], horizontal=True, key=f"{prefix}_bmode")
    beta = est.params["beta"]
    if mode == "Valor livre":
        beta = c2.slider("β", 1.0, 9.0, float(round(beta, 2)), 0.05, key=f"{prefix}_beta")
    else:
        c2.markdown(f"β = {beta:.4f} (A_s = {est.params['As_design_dB']:.2f} dB)")
    h = ideal_response(spec, est.cutoffs, M) * W.kaiser(M, beta)
    devs = fa.band_deviations(h, spec.bands, fa.frequency_grid(spec.edges, fa.DENSE_GRID_POINTS))
    rows = ["| Faixa | Ganho | Desvio máximo | Tolerância | Situação |", "|---|---|---|---|---|"]
    for b, dv in zip(spec.bands, devs):
        rows.append(f"| [{b.lo:g}π, {b.hi:g}π] | {b.gain:g} | {dv:.5f} | {b.delta:g} | {status_badge(dv <= b.delta)} |")
    st.markdown("\n".join(rows), unsafe_allow_html=True)
    dmin = min(b.delta for b in spec.bands)
    fig = figures.overlay_db({"curves": {f"explorado: M = {M}, β = {beta:.3f}": h, f"oficial: M = {final.M}": final.h},
                              "xlim": (0, 1), "ylim": (-100, 5), "limit_db": 20 * np.log10(dmin),
                              "limit_label": f"20·log10(δ_min) = {20 * np.log10(dmin):.1f} dB", "edges": list(spec.edges)})
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_fig")


def iir_order(objs, prefix: str, n_range=None) -> None:
    """Ordem N do protótipo, mantendo o critério de R_p exato em Ω_p."""
    spec, official = objs["spec"], objs["proto"]
    lo, hi = n_range or (max(1, official.N - 4), official.N + 5)
    N = st.slider("Ordem N", lo, hi, official.N, key=f"{prefix}_N")
    proto = OF_ORDER[official.approximation](N, spec.wp, spec.rp_db)
    att = -ia.magnitude_db(ia.frequency_response(proto, np.array([spec.ws])))[0]
    unit = "kHz" if spec.freq_unit == "Hz" else "rad/s"
    ws_disp = spec.ws / 2 / np.pi / 1e3 if spec.freq_unit == "Hz" else spec.ws
    st.markdown(f"Atenuação na borda de rejeição ({ws_disp:g} {unit}): **{att:.2f} dB** "
                f"(exigido ≥ {spec.as_db:g} dB) &nbsp; {status_badge(att >= spec.as_db)}", unsafe_allow_html=True)
    w = np.linspace(0, 2 * spec.ws, 4001)
    fig = figures.analog_db(official, spec, w)
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(np.abs(ia.frequency_response(proto, w)))
    x = w / 2 / np.pi if spec.freq_unit == "Hz" else w
    fig.data[0].name = f"oficial N = {official.N}"
    fig.add_trace(go.Scatter(x=x, y=db, mode="lines", name=f"explorado N = {N}",
                             line=dict(color=figures.SERIES2, width=2, dash="dash")))
    fig.update_layout(showlegend=True, margin=dict(t=60), legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_fig")
