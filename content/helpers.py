"""Funções auxiliares comuns aos módulos de conteúdo (cálculo de contexto e tabelas).

Nada aqui é texto didático: só preparação de valores vindos do núcleo, para
que os módulos ``content/pX_YY.py`` fiquem focados na explicação.
"""

from __future__ import annotations

import math

import numpy as np

from dsp_av2.fir import analysis as fa
from dsp_av2.iir import analysis as ia

TWO_PI = 2 * math.pi


def md_table(headers, rows) -> str:
    """Tabela Markdown (células já formatadas; sem barras verticais internas)."""
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ok(flag: bool) -> str:
    return "✓ sim" if flag else "✗ não"


def fir_db_measure(h, spec, n=fa.DENSE_GRID_POINTS):
    """(Rp, As) medidos na convenção do livro (dB normalizado), grade com bordas."""
    w = fa.frequency_grid(spec.edges, n)
    return fa.measure_rp_as(h, spec.passbands, spec.stopbands, w)


def fir_db_context(spec, d, report) -> dict:
    """Valores comuns aos FIR especificados em dB (P7.9–P7.12, P7.16)."""
    rp_book, as_book = report.checks[0].measured, report.checks[1].measured
    rp, as_ = report.checks[2].measured, report.checks[3].measured
    ctx = {
        "Rp_spec": spec.rp_db, "As_spec": spec.as_db,
        "dw": d.transition_width, "M_table": d.M_table, "M": d.M, "alpha": d.alpha,
        "type": report.linear_phase_type,
        "rp": rp, "as_": as_, "rp_book": rp_book, "as_book": as_book,
        "rp_margin": spec.rp_db - rp, "as_margin": as_ - spec.as_db,
        "n_grid": fa.DENSE_GRID_POINTS, "n_book": fa.BOOK_GRID_POINTS,
        "delta1_meas": (10 ** (rp / 20) - 1) / (10 ** (rp / 20) + 1),
        "delta2_meas": 10 ** (-as_ / 20),
    }
    for i, c in enumerate(d.cutoffs, 1):
        ctx[f"wc{i}"] = c
    for i, b in enumerate(spec.bands, 1):
        ctx[f"b{i}_lo"], ctx[f"b{i}_hi"] = b.lo, b.hi
    for i, t in enumerate(d.transitions, 1):
        ctx[f"t{i}"] = t.width
    return ctx


def worst_stop_point(h, spec, n=200001):
    """Frequência (×π) e nível (dB normalizado) do pior ponto da rejeição."""
    w = fa.frequency_grid(spec.edges, n)
    db = fa.magnitude_db(fa.frequency_response(h, w))
    mask = np.zeros_like(w, dtype=bool)
    for b in spec.stopbands:
        mask |= (w >= b.lo * np.pi - 1e-12) & (w <= b.hi * np.pi + 1e-12)
    i = np.argmax(np.where(mask, db, -np.inf))
    return w[i] / np.pi, db[i]


def iir_edges_context(spec, proto) -> dict:
    """Perda em Ωp, atenuação em Ωs, frequências e ganho DC de um protótipo analógico."""
    loss_wp = -ia.magnitude_db(ia.frequency_response(proto, np.array([spec.wp])))[0]
    att_ws = -ia.magnitude_db(ia.frequency_response(proto, np.array([spec.ws])))[0]
    h0 = abs(ia.frequency_response(proto, np.array([0.0]))[0])
    return {
        "Rp": spec.rp_db, "As": spec.as_db, "Rp10": spec.rp_db / 10, "As10": spec.as_db / 10,
        "Wp": spec.wp, "Ws": spec.ws, "Fp": spec.wp / TWO_PI, "Fs": spec.ws / TWO_PI,
        "Fp_k": spec.wp / TWO_PI / 1e3, "Fs_k": spec.ws / TWO_PI / 1e3, "ratio": spec.wp / spec.ws,
        "Wr": spec.ws / spec.wp,
        "N": proto.N, "N_real": proto.params["N_real"], "Nm1": proto.N - 1, "N2": 2 * proto.N,
        "loss_wp": loss_wp, "att_ws": att_ws, "margin_ws": att_ws - spec.as_db,
        "H0": h0, "H0_db": 20 * math.log10(h0),
    }


def pole_table(proto, unit_khz: bool = False) -> str:
    rows = []
    for k, p in enumerate(proto.p):
        ang = np.degrees(np.angle(p)) % 360
        if unit_khz:
            val = f"${p.real:,.1f} {'+' if p.imag >= 0 else '-'} j{abs(p.imag):,.1f}$".replace(",", "\\,")
        else:
            val = f"${p.real:.4f} {'+' if p.imag >= 0 else '-'} j{abs(p.imag):.4f}$"
        mag = f"${abs(p):,.2f}$".replace(",", "\\,") if unit_khz else f"{abs(p):.4f}"
        rows.append((k, f"{ang:.1f}°", mag, val))
    return md_table(["$k$", "ângulo de $p_k$", "$\\lvert p_k\\rvert$", "$p_k$ (rad/s)"], rows)


def group_delay_stats(proto, w_max) -> dict:
    w = np.linspace(0, w_max, 40001)
    gd = ia.group_delay(proto, w)
    gd0 = ia.group_delay(proto, np.array([0.0]))[0]
    return {"gd0": gd0, "gd_peak": gd.max(), "gd_peak_w": w[np.argmax(gd)]}
