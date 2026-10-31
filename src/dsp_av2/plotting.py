"""Visualização dos resultados. Nenhum cálculo de projeto é feito aqui.

As funções recebem dados já calculados e devolvem ``matplotlib.figure.Figure``.
Estilo: uma série por gráfico (azul), limites da especificação em cinza
tracejado com rótulo direto, grade discreta.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

SERIES = "#2a78d6"
LIMIT = "#52514e"
TEXT = "#0b0b0b"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LIMIT)
    ax.tick_params(colors=LIMIT, labelcolor=TEXT)


def _limit_line(ax, y: float, label: str) -> None:
    ax.axhline(y, color=LIMIT, linestyle="--", linewidth=1)
    ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"),
                xytext=(-4, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=8, color=LIMIT)


def plot_impulse_response(ax, h: np.ndarray, title: str = "Resposta ao impulso h(n)") -> None:
    n = np.arange(len(h))
    markers, stems, base = ax.stem(n, h, basefmt=" ")
    plt.setp(stems, color=SERIES, linewidth=1)
    plt.setp(markers, color=SERIES, markersize=3)
    ax.axhline(0, color=LIMIT, linewidth=0.8)
    alpha = (len(h) - 1) / 2
    ax.axvline(alpha, color=LIMIT, linestyle=":", linewidth=1)
    ax.annotate(f"α = {alpha:g}", xy=(alpha, 0.0), xycoords=("data", "axes fraction"),
                xytext=(4, 4), textcoords="offset points", fontsize=8, color=LIMIT)
    ax.set_xlim(-1, len(h))
    ax.set_xlabel("n")
    ax.set_ylabel("h(n)")
    ax.set_title(title, loc="left", fontsize=10)
    _style(ax)


def plot_magnitude_db(ax, w: np.ndarray, db: np.ndarray, edges_pi=(), rp_db=None, as_db=None,
                      ylim=(-100, 5), title: str = "Magnitude em dB") -> None:
    """|H| em dB vs ω/π, com bordas (verticais) e limites -R_p, -A_s."""
    ax.plot(w / np.pi, db, color=SERIES, linewidth=1.5)
    for e in edges_pi:
        ax.axvline(e, color=LIMIT, linestyle=":", linewidth=1)
    if rp_db is not None:
        _limit_line(ax, -rp_db, f"−Rp = −{rp_db:.4g} dB")
    if as_db is not None:
        _limit_line(ax, -as_db, f"−As = −{as_db:.4g} dB")
    ticks = sorted({0.0, 1.0, *edges_pi})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    ax.set_xlabel("ω / π")
    ax.set_ylabel("|H| (dB)")
    ax.set_title(title, loc="left", fontsize=10)
    _style(ax)


def plot_passband_zoom(ax, w, db, passbands, rp_db, title: str = "Zoom na passagem (verificação de Rp)") -> None:
    """|H| em dB restrita às faixas de passagem, escala vertical em décimos de dB."""
    for pb in passbands:
        m = (w >= pb.lo * np.pi - 1e-12) & (w <= pb.hi * np.pi + 1e-12)
        ax.plot(w[m] / np.pi, db[m], color=SERIES, linewidth=1.5)
    _limit_line(ax, -rp_db, f"−Rp = −{rp_db:.4g} dB")
    ax.axhline(0, color=LIMIT, linewidth=0.8)
    lo = min(pb.lo for pb in passbands)
    hi = max(pb.hi for pb in passbands)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-1.5 * rp_db, 0.25 * rp_db)
    ax.set_xlabel("ω / π")
    ax.set_ylabel("|H| (dB)")
    ax.set_title(title, loc="left", fontsize=10)
    _style(ax)


def plot_amplitude_response(ax, w, Hr, bands, title: str = "Resposta de amplitude Hr(ω)") -> None:
    """H_r(ω) real (com sinal), com os limites G ± δ de cada faixa da especificação."""
    ax.plot(w / np.pi, Hr, color=SERIES, linewidth=1.5)
    for b in bands:
        for y in (b.gain - b.delta, b.gain + b.delta):
            ax.hlines(y, b.lo, b.hi, colors=LIMIT, linestyles="--", linewidth=1)
    edges = sorted({x for b in bands for x in (b.lo, b.hi)})
    for e in edges:
        if 0 < e < 1:
            ax.axvline(e, color=LIMIT, linestyle=":", linewidth=1)
    ax.set_xticks(edges)
    ax.set_xticklabels([f"{t:g}" for t in edges], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("ω / π")
    ax.set_ylabel("Hr(ω)")
    ax.set_title(title + "  (tracejado: G ± δ)", loc="left", fontsize=10)
    _style(ax)


def fir_tolerance_figure(h, w, Hr, db, bands, as_db, suptitle: str, panels=("impulse", "amplitude", "db")):
    """Figura FIR para especificações por tolerância (Kaiser): h(n), Hr(ω) e |H| dB."""
    edges = tuple(sorted({x for b in bands for x in (b.lo, b.hi)} - {0.0, 1.0}))
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 3.5 * len(panels)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")
    for ax, panel in zip(axes, panels):
        if panel == "impulse":
            plot_impulse_response(ax, h)
        elif panel == "amplitude":
            plot_amplitude_response(ax, w, Hr, bands)
        elif panel == "db":
            plot_magnitude_db(ax, w, db, edges, as_db=as_db)
    fig.suptitle(suptitle, fontsize=12, x=0.02, ha="left")
    return fig


def fir_overview_figure(h, w, db, edges_pi, rp_db, as_db, suptitle: str, passbands=()):
    """Figura padrão FIR: h(n) e |H| em dB (o que P7.9–P7.11 pedem).

    Se ``passbands`` for dado, acrescenta um painel complementar com zoom na
    passagem, onde a linha −R_p fica visível.
    """
    rows = 3 if passbands else 2
    fig, axes = plt.subplots(rows, 1, figsize=(8, 3.5 * rows), constrained_layout=True)
    fig.patch.set_facecolor("white")
    plot_impulse_response(axes[0], h)
    plot_magnitude_db(axes[1], w, db, edges_pi, rp_db, as_db)
    if passbands:
        plot_passband_zoom(axes[2], w, db, passbands, rp_db)
    fig.suptitle(suptitle, fontsize=12, x=0.02, ha="left")
    return fig


# ---------------------------------------------------------------------------
# Protótipos analógicos (Capítulo 8)
# ---------------------------------------------------------------------------

def _freq_axis(w: np.ndarray, unit: str) -> tuple[np.ndarray, str]:
    """Converte rad/s para a unidade de exibição (decisão D9)."""
    if unit == "Hz":
        return w / (2 * np.pi), "Frequência (Hz)"
    return w, "Ω (rad/s)"


def _time_scale(t_max: float) -> tuple[float, str]:
    """Escala de exibição para tempos: ms abaixo de 0.1 s (evita eixos 0.0001…)."""
    return (1e3, "ms") if t_max < 0.1 else (1.0, "s")


def _edge_lines(ax, edges, unit: str) -> None:
    labels = ("Fp", "Fs") if unit == "Hz" else ("Ωp", "Ωs")
    for e, lab in zip(edges, labels):
        x = e / (2 * np.pi) if unit == "Hz" else e
        ax.axvline(x, color=LIMIT, linestyle=":", linewidth=1)
        ax.annotate(lab, xy=(x, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(3, 4), textcoords="offset points", fontsize=8, color=LIMIT)


def analog_overview_figure(*, w, H, panels, edges, unit, rp_db, as_db, suptitle,
                           phase=None, gd=None, t=None, h=None):
    """Figura 2×2 com os painéis pedidos pelo enunciado.

    ``panels`` ⊂ {"magnitude", "magnitude_db", "phase", "group_delay", "impulse"}.
    Todos os dados chegam prontos; aqui só se desenha.
    """
    x, xlabel = _freq_axis(w, unit)
    mag = np.abs(H)
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(mag)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax, panel in zip(axes.flat, panels):
        if panel == "magnitude":
            ax.plot(x, mag, color=SERIES, linewidth=1.5)
            _limit_line(ax, 10 ** (-rp_db / 20), f"10^(−Rp/20) = {10 ** (-rp_db / 20):.4f}")
            _limit_line(ax, 10 ** (-as_db / 20), f"10^(−As/20) = {10 ** (-as_db / 20):.1e}")
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("|H(jΩ)|")
            ax.set_title("Magnitude", loc="left", fontsize=10)
        elif panel == "magnitude_db":
            ax.plot(x, db, color=SERIES, linewidth=1.5)
            _limit_line(ax, -rp_db, f"−Rp = −{rp_db:.4g} dB")
            _limit_line(ax, -as_db, f"−As = −{as_db:.4g} dB")
            ax.set_ylim(-as_db - 30, 5)
            ax.set_ylabel("|H(jΩ)| (dB)")
            ax.set_title("Log-magnitude", loc="left", fontsize=10)
        elif panel == "phase":
            # Fase principal (−π, π], como a freqs_m do livro; NaN nos saltos
            # de 2π evita desenhar a descontinuidade como um traço vertical.
            ph = phase / np.pi
            ph = np.where(np.abs(np.diff(ph, prepend=ph[0])) > 1.0, np.nan, ph)
            ax.plot(x, ph, color=SERIES, linewidth=1.5)
            ax.set_ylim(-1.05, 1.05)
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_ylabel("∠H(jΩ) (unidades de π rad)")
            ax.set_title("Fase", loc="left", fontsize=10)
        elif panel == "group_delay":
            k, tu = _time_scale(np.max(gd))
            ax.plot(x, k * gd, color=SERIES, linewidth=1.5)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(f"τ(Ω) ({tu})")
            ax.set_title("Atraso de grupo", loc="left", fontsize=10)
        elif panel == "impulse":
            k, tu = _time_scale(t[-1])
            ax.plot(k * t, h, color=SERIES, linewidth=1.5)
            ax.axhline(0, color=LIMIT, linewidth=0.8)
            ax.set_xlim(k * t[0], k * t[-1])
            ax.set_xlabel(f"t ({tu})")
            ax.set_ylabel("h(t)")
            ax.set_title("Resposta ao impulso", loc="left", fontsize=10)
            _style(ax)
            continue
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel(xlabel)
        _edge_lines(ax, edges, unit)
        _style(ax)
    fig.suptitle(suptitle, fontsize=12, x=0.02, ha="left")
    return fig
