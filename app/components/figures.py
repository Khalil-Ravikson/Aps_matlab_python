"""Figuras Plotly da aplicação. Só desenham: os dados vêm prontos do núcleo.

As mesmas grandezas aparecem no relatório PDF em Matplotlib (``plotting.py``);
aqui a vantagem é a leitura de valores ao passar o mouse e o zoom.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from dsp_av2.fir import analysis as fa
from dsp_av2.iir import analysis as ia

SERIES = "#2a78d6"
SERIES2 = "#eb6834"
LIMIT = "#5a5953"
BAND = "rgba(15,122,90,0.10)"
GRID = "#ebe9e3"
FONT = dict(family="Source Sans Pro, sans-serif", size=13, color="#1c1c1a")


def _base(fig: go.Figure, xlabel: str, ylabel: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=60, r=20, t=30, b=50), font=FONT,
        plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        showlegend=False, legend=dict(orientation="h", y=1.08, x=0),
    )
    fig.update_xaxes(title=xlabel, gridcolor=GRID, zeroline=False, linecolor=LIMIT, ticks="outside")
    fig.update_yaxes(title=ylabel, gridcolor=GRID, zeroline=False, linecolor=LIMIT, ticks="outside")
    return fig


def _hline(fig, y, text):
    fig.add_hline(y=y, line=dict(color=LIMIT, dash="dash", width=1),
                  annotation_text=text, annotation_position="top right",
                  annotation_font=dict(size=11, color=LIMIT))


def _vlines(fig, xs, labels=None):
    for i, x in enumerate(xs):
        fig.add_vline(x=x, line=dict(color=LIMIT, dash="dot", width=1))
        if labels:
            fig.add_annotation(x=x, y=0, yref="paper", text=labels[i], showarrow=False,
                               yshift=10, xshift=14, font=dict(size=11, color=LIMIT))


# ── FIR ─────────────────────────────────────────────────────────────────────

def fir_impulse(h: np.ndarray) -> go.Figure:
    n = np.arange(len(h))
    alpha = (len(h) - 1) / 2
    fig = go.Figure()
    xs = np.repeat(n, 3).astype(float)
    ys = np.column_stack([np.zeros_like(h), h, np.full_like(h, np.nan)]).ravel()
    xs[2::3] = np.nan
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=SERIES, width=1.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=n, y=h, mode="markers", marker=dict(color=SERIES, size=7),
                             hovertemplate="n = %{x}<br>h(n) = %{y:.5f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=LIMIT, width=1))
    _vlines(fig, [alpha], [f"α = {alpha:g}"])
    fig.update_layout(hovermode="closest")
    return _base(fig, "n", "h(n)")


def fir_amplitude(w: np.ndarray, Hr: np.ndarray, bands) -> go.Figure:
    fig = go.Figure()
    for b in bands:
        fig.add_shape(type="rect", x0=b.lo, x1=b.hi, y0=b.gain - b.delta, y1=b.gain + b.delta,
                      fillcolor=BAND, line=dict(width=0), layer="below")
        for y in (b.gain - b.delta, b.gain + b.delta):
            fig.add_shape(type="line", x0=b.lo, x1=b.hi, y0=y, y1=y, line=dict(color=LIMIT, dash="dash", width=1))
    fig.add_trace(go.Scatter(x=w / np.pi, y=Hr, mode="lines", line=dict(color=SERIES, width=2),
                             hovertemplate="ω = %{x:.4f}π<br>Hr = %{y:.5f}<extra></extra>"))
    _vlines(fig, sorted({x for b in bands for x in (b.lo, b.hi)} - {0.0, 1.0}))
    return _base(fig, "ω / π", "Hr(ω)")


def fir_db(w: np.ndarray, db: np.ndarray, edges, as_db: float, rp_db: float | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w / np.pi, y=db, mode="lines", line=dict(color=SERIES, width=2),
                             hovertemplate="ω = %{x:.4f}π<br>|H| = %{y:.2f} dB<extra></extra>"))
    _hline(fig, -as_db, f"−As = −{as_db:.4g} dB")
    if rp_db is not None:
        _hline(fig, -rp_db, f"−Rp = −{rp_db:.4g} dB")
    _vlines(fig, edges)
    fig.update_yaxes(range=[-100, 5])
    return _base(fig, "ω / π", "|H| (dB)")


def fir_compare_stopband(w: np.ndarray, designs: dict[str, np.ndarray], band, limit: float) -> go.Figure:
    """|H| linear na faixa de rejeição para vários projetos (ex.: estimativa × mínimo)."""
    fig = go.Figure()
    m = (w >= band.lo * np.pi - 0.03 * np.pi) & (w <= band.hi * np.pi + 0.03 * np.pi)
    for (label, h), color in zip(designs.items(), (SERIES2, SERIES)):
        mag = np.abs(fa.frequency_response(h, w[m]))
        fig.add_trace(go.Scatter(x=w[m] / np.pi, y=mag, mode="lines", name=label, line=dict(color=color, width=2),
                                 hovertemplate=f"{label}<br>ω = %{{x:.4f}}π<br>|H| = %{{y:.5f}}<extra></extra>"))
    _hline(fig, limit, f"limite δ₂ = {limit:g}")
    _vlines(fig, [band.lo, band.hi])
    fig.update_yaxes(range=[0, 2.2 * limit])
    fig = _base(fig, "ω / π", "|H(e^jω)|")
    fig.update_layout(showlegend=True, margin=dict(t=60), legend=dict(orientation="h", y=1.12, x=0))
    return fig


# ── IIR (protótipos analógicos) ─────────────────────────────────────────────

def _fx(w, unit):
    return (w / (2 * np.pi), "Frequência (Hz)") if unit == "Hz" else (w, "Ω (rad/s)")


def _edges(fig, spec):
    unit = spec.freq_unit
    xs = [spec.wp, spec.ws]
    if unit == "Hz":
        xs = [x / (2 * np.pi) for x in xs]
    _vlines(fig, xs, ["Fp", "Fs"] if unit == "Hz" else ["Ωp", "Ωs"])


def analog_magnitude(proto, spec, w) -> go.Figure:
    x, xl = _fx(w, spec.freq_unit)
    mag = np.abs(ia.frequency_response(proto, w))
    fig = go.Figure(go.Scatter(x=x, y=mag, mode="lines", line=dict(color=SERIES, width=2),
                               hovertemplate="%{x:.1f}<br>|H| = %{y:.5f}<extra></extra>"))
    _hline(fig, 10 ** (-spec.rp_db / 20), f"10^(−Rp/20) = {10 ** (-spec.rp_db / 20):.4f}")
    _edges(fig, spec)
    fig.update_yaxes(range=[0, 1.1])
    return _base(fig, xl, "|H(jΩ)|")


def analog_db(proto, spec, w) -> go.Figure:
    x, xl = _fx(w, spec.freq_unit)
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(np.abs(ia.frequency_response(proto, w)))
    fig = go.Figure(go.Scatter(x=x, y=db, mode="lines", line=dict(color=SERIES, width=2),
                               hovertemplate="%{x:.1f}<br>|H| = %{y:.2f} dB<extra></extra>"))
    _hline(fig, -spec.rp_db, f"−Rp = −{spec.rp_db:g} dB")
    _hline(fig, -spec.as_db, f"−As = −{spec.as_db:g} dB")
    _edges(fig, spec)
    fig.update_yaxes(range=[-spec.as_db - 30, 5])
    return _base(fig, xl, "|H(jΩ)| (dB)")


def analog_group_delay(proto, spec, w) -> go.Figure:
    x, xl = _fx(w, spec.freq_unit)
    tau = ia.group_delay(proto, w)
    k, tu = (1e3, "ms") if tau.max() < 0.1 else (1.0, "s")
    fig = go.Figure(go.Scatter(x=x, y=k * tau, mode="lines", line=dict(color=SERIES, width=2),
                               hovertemplate=f"%{{x:.1f}}<br>τ = %{{y:.4f}} {tu}<extra></extra>"))
    _edges(fig, spec)
    fig.update_yaxes(rangemode="tozero")
    return _base(fig, xl, f"τ(Ω) ({tu})")


def analog_phase(proto, spec, w) -> go.Figure:
    x, xl = _fx(w, spec.freq_unit)
    ph = np.angle(ia.frequency_response(proto, w)) / np.pi
    ph = np.where(np.abs(np.diff(ph, prepend=ph[0])) > 1.0, np.nan, ph)
    fig = go.Figure(go.Scatter(x=x, y=ph, mode="lines", line=dict(color=SERIES, width=2),
                               hovertemplate="%{x:.1f}<br>∠H = %{y:.3f}π<extra></extra>"))
    _edges(fig, spec)
    fig.update_yaxes(range=[-1.05, 1.05])
    return _base(fig, xl, "∠H(jΩ) (unidades de π)")


def analog_impulse(proto) -> go.Figure:
    t = np.linspace(0, ia.impulse_time_span(proto), 2001)
    h = ia.impulse_response(proto, t)
    k, tu = (1e3, "ms") if t[-1] < 0.1 else (1.0, "s")
    fig = go.Figure(go.Scatter(x=k * t, y=h, mode="lines", line=dict(color=SERIES, width=2),
                               hovertemplate=f"t = %{{x:.4f}} {tu}<br>h(t) = %{{y:.2f}}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=LIMIT, width=1))
    return _base(fig, f"t ({tu})", "h(t)")


def s_plane(proto) -> go.Figure:
    """Polos no plano s, com o círculo de raio |p| e ângulos a partir do eixo real negativo."""
    p = np.asarray(proto.p)
    r = np.max(np.abs(p))
    th = np.linspace(np.pi / 2, 3 * np.pi / 2, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r * np.cos(th), y=r * np.sin(th), mode="lines",
                             line=dict(color=LIMIT, dash="dot", width=1), hoverinfo="skip"))
    for q in p:
        fig.add_trace(go.Scatter(x=[0, q.real], y=[0, q.imag], mode="lines",
                                 line=dict(color=GRID, width=1), hoverinfo="skip"))
    ang = 180 - np.abs(np.degrees(np.angle(p)))
    fig.add_trace(go.Scatter(x=p.real, y=p.imag, mode="markers", marker=dict(symbol="x", size=12, color=SERIES, line=dict(width=2)),
                             customdata=ang,
                             hovertemplate="σ = %{x:.1f}<br>jω = %{y:.1f}<br>θ = %{customdata:.0f}° do eixo real negativo<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=LIMIT, width=1))
    fig.add_hline(y=0, line=dict(color=LIMIT, width=1))
    fig.update_layout(hovermode="closest")
    fig = _base(fig, "σ = Re(s)  (rad/s)", "jΩ = Im(s)  (rad/s)", height=440)
    fig.update_xaxes(range=[-1.15 * r, 0.3 * r])
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[-1.1 * r, 1.1 * r])
    return fig


# ── Mapeamento exercício → figuras ──────────────────────────────────────────

def build(ex_id: str, key: str, objs: dict) -> go.Figure:
    if ex_id in ("p7_13",):
        spec, d = objs["spec"], objs["final"]
        w = fa.frequency_grid(spec.edges, fa.DENSE_GRID_POINTS)
        if key == "impulse":
            return fir_impulse(d.h)
        if key == "amplitude":
            return fir_amplitude(w, fa.amplitude_response(d.h, w), spec.bands)
        if key == "db":
            with np.errstate(divide="ignore"):
                db = 20 * np.log10(np.abs(fa.frequency_response(d.h, w)))
            return fir_db(w, db, spec.edges, d.params["As_design_dB"])
        if key == "compare":
            est = objs["estimate"]
            return fir_compare_stopband(w, {f"estimativa M = {est.M}": est.h, f"mínimo M = {d.M}": d.h},
                                        spec.stopbands[0], spec.stopbands[0].delta)
    if ex_id in ("p8_13",):
        spec, proto = objs["spec"], objs["proto"]
        w = np.linspace(0, 2 * spec.ws, 4001)
        return {
            "poles": lambda: s_plane(proto),
            "magnitude": lambda: analog_magnitude(proto, spec, w),
            "db": lambda: analog_db(proto, spec, w),
            "group_delay": lambda: analog_group_delay(proto, spec, w),
            "phase": lambda: analog_phase(proto, spec, w),
            "impulse": lambda: analog_impulse(proto),
        }[key]()
    raise KeyError(f"figura não definida: {ex_id}/{key}")
