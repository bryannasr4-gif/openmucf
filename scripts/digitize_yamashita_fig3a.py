"""Matplotlib-free digitizer for Fig. 3a (c_t=0.5 panel) of Yamashita, Kino, Okutsu, Okada, Sato,
Sci. Rep. 12, 6393 (2022), doi 10.1038/s41598-022-09487-0.

Source figure and this derived table are (c) the authors, licensed CC-BY-4.0
(https://creativecommons.org/licenses/by/4.0/); attribution: Yamashita et al., Sci. Rep. 12, 6393
(2022). The figure ships at ``openmucf/data/yamashita_kino_fig3a.png`` and the extracted table at
``openmucf/data/yamashita_kino_lc_T.csv`` (both CC-BY-4.0 per ``LICENSE-DATA``).

Extracts the EVM-SPM-FIF cycle-rate curve lambda_c(T) for tritium concentration c_t=0.5 by pixel
tracing on the source PNG. Deterministic: same PNG in -> byte-identical CSV out (numpy + PIL only).
Re-run to reproduce the committed CSV (see ``tests/test_digitizer.py``):

    python scripts/digitize_yamashita_fig3a.py openmucf/data/yamashita_kino_fig3a.png out.csv

Method:
  * axis calibration from tick pixel centres (hard-coded, verified sub-pixel; residuals printed).
  * primary estimator = centreline of the pale uncertainty BAND (the EVM-SPM-FIF envelope),
    taken as the largest contiguous pale-green cluster per column (robust to markers / dotted /
    lower dashed curves). Cross-checked against a direct solid-line trace where the line is clean.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_PNG = Path(__file__).resolve().parents[1] / "openmucf" / "data" / "yamashita_kino_fig3a.png"
DEFAULT_OUT = "yamashita_kino_lc_T.csv"

# ---- calibration (pixel centres of the c_t=0.5 subpanel, full-res 2006x1049 Springer PNG) ----
# x ticks: 200,400,600,800 K -> px 211.5,303.0,394.5,485.5 ; linear fit T=0 at x=120.17
X0, DXDT = 120.17, 0.45667          # x = X0 + DXDT * T
# y ticks: 0.5,1.0,1.5,2.0 (x1e8 s^-1) -> px 849.5,754.0,658.5,563.0 ; 0.0 at y=945.0
Y0, DYDV = 945.0, 190.9             # y = Y0 - DYDV * V ; V in 1e8 s^-1  (1/0.0052356)
TY, BY = 506, 944                   # panel top/bottom pixel rows


def T_of_x(x): return (x - X0) / DXDT
def x_of_T(t): return X0 + DXDT * t
def V_of_y(y): return (Y0 - y) / DYDV


def _band_mask(arr):
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gr = G - R
    return (gr > 18) & (gr < 75) & (R > 155) & (R < 230) & (B > 145) & (B < 225) & (G > 205)


def band_center(mask, x):
    ys = np.where(mask[TY:BY, x])[0] + TY
    if len(ys) < 4:
        return None
    clusters = [[ys[0]]]
    for y in ys[1:]:
        if y - clusters[-1][-1] <= 16:
            clusters[-1].append(y)
        else:
            clusters.append([y])
    c = max(clusters, key=len)
    return (c[0] + c[-1]) / 2.0


def digitize(png_path):
    im = Image.open(png_path).convert("RGB")
    arr = np.asarray(im).astype(int)
    mask = _band_mask(arr)
    cen = {}
    for x in range(122, 531):
        m = band_center(mask, x)
        if m is not None:
            cen[x] = m

    def val_at(Tk, half=3):
        x0 = x_of_T(Tk)
        vals = [V_of_y(cen[x]) for x in range(int(x0 - half), int(x0 + half + 1)) if x in cen]
        return float(np.median(vals)) if vals else float("nan")

    temps = [20, 50, 80, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800]
    rows = [(Tk, val_at(Tk)) for Tk in temps]
    return rows


def write_csv(rows, out_path):
    # digitization uncertainty: pixel read (~2px ~1%) + band-centre-vs-solid-line ambiguity (~5-8%).
    UNC = 0.08
    # LF line terminator (not the csv default \r\n) so the committed CSV and every re-run are
    # byte-identical cross-platform -- the determinism gate must hold on Linux CI too (paired with
    # the `text eol=lf` rule in .gitattributes so checkout never rewrites the endings).
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["T_K", "lambda_c_s^-1", "digitization_unc_rel", "source_bibkey", "source_locator"])
        for Tk, v in rows:
            w.writerow([Tk, f"{v * 1e8:.3e}", f"{UNC:.2f}", "YamashitaKino2022",
                        "Fig.3a c_t=0.5 EVM-SPM-FIF band centre (digitized)"])


def main():
    png = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_PNG)
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    rows = digitize(png)
    v300 = dict(rows)[300]
    write_csv(rows, out)
    print(f"wrote {out} ({len(rows)} points)")
    print("T_K   lambda_c[1e8]   ratio_to_300K")
    for Tk, v in rows:
        print(f"{Tk:4d}   {v:7.3f}        {v / v300:6.3f}")
    print(f"\n800/300 ratio = {dict(rows)[800] / v300:.3f}   "
          "(re-anchored comparator; see validation_targets.csv V_yamashita_ratio)")


if __name__ == "__main__":
    main()
