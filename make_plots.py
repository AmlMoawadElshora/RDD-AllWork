"""Publication-ready plots for the RapierNet road-damage-detection paper.

Reads an Ultralytics YOLO results.csv and produces three 300 DPI figures
(loss curves, metric convergence, StripRFNet-vs-RapierNet comparison) as
both PDF (vector) and PNG. Style is IEEE/Springer: serif font, clean
gridlines, minimal spines, distinct professional colors.

Usage:
    python make_plots.py                     # uses ./results.csv, writes ./plots
    python make_plots.py -r /kaggle/working/rapi5/results.csv -o /kaggle/working/plots
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except ImportError:
    sns = None

# ----------------------------------------------------------------------------- style
_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#b9b9b9",
    "grid.linestyle": "-",
    "grid.alpha": 0.35,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}
matplotlib.rcParams.update(_RC)
if sns is not None:
    sns.set_style("whitegrid", {"font.family": "serif", "axes.grid": False,
                                "axes.spines.left": True, "axes.spines.bottom": True})

_PALETTE = ["#0b5394", "#c00000", "#2e7d32", "#7b2fbe",
            "#e0810d", "#0f7c8c", "#98a103", "#8c564b"]

COL_TRAIN_BOX = "train/box_loss"
COL_TRAIN_CLS = "train/cls_loss"
COL_TRAIN_DFL = "train/dfl_loss"
COL_VAL_BOX = "val/box_loss"
COL_VAL_CLS = "val/cls_loss"
COL_VAL_DFL = "val/dfl_loss"
COL_PREC = "metrics/precision(B)"
COL_RECALL = "metrics/recall(B)"
COL_MAP50 = "metrics/mAP50(B)"
COL_MAP50_95 = "metrics/mAP50-95(B)"

PROTO_MAP50 = 0.838
PROTO_MAP50_95 = 0.540
PROTO_F1 = 0.841


def _pick(df: pd.DataFrame, *names: str) -> str | None:
    """"Return the first column of df matching any of names (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        hit = lower.get(n.lower())
        if hit is not None:
            return hit
    return None


def find_results() -> str:
    """Auto-locate a results.csv: cwd first, then common Kaggle paths."""
    candidates = ["results.csv",
                  "/kaggle/working/rapi5/results.csv",
                  "/kaggle/working/rapiernet/results.csv",
                  "/kaggle/working/rapi5_hr/results.csv"]
    for c in candidates:
        if os.path.isfile(c):
            return c
    if os.path.isdir("/kaggle/working"):
        import glob as _glob
        hits = sorted(_glob.glob("/kaggle/working/**/results.csv", recursive=True))
        if hits:
            return hits[0]
    return ""


def load_results(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"results.csv not found: {csv_path}\n"
            "  Run this script where the training output lives, e.g. "
            "/kaggle/working/rapi5/results.csv (or pass -r).")
    df = pd.read_csv(csv_path)
    ep = _pick(df, "epoch")
    if ep is not None:
        df = df[pd.to_numeric(df[ep], errors="coerce").notna()].copy()
        df[ep] = df[ep].astype(float)
    else:
        df.insert(0, "epoch", np.arange(len(df)))
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.empty:
        raise ValueError(f"{csv_path} contains no usable rows.")
    return df


def _epochs(df: pd.DataFrame) -> np.ndarray:
    ep = _pick(df, "epoch")
    return df[ep].to_numpy()


def save(fig: plt.Figure, out_dir: str, name: str) -> None:
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


def fig_losses(df: pd.DataFrame, out_dir: str) -> None:
    """Figure 1 - training vs validation loss curves (3 panels)."""
    epochs = _epochs(df)
    pairs = [
        ("Box loss", COL_TRAIN_BOX, COL_VAL_BOX),
        ("Classification loss", COL_TRAIN_CLS, COL_VAL_CLS),
        ("DFL loss", COL_TRAIN_DFL, COL_VAL_DFL),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))
    for ax, (title, tr, vl) in zip(axes, pairs):
        trc = _pick(df, tr)
        if trc is None:
            ax.set_visible(False)
            continue
        ax.plot(epochs, df[trc].to_numpy(), lw=1.4, color=_PALETTE[0],
                label="Train")
        if vl is not None:
            vlc = _pick(df, vl)
            if vlc is not None and df[vlc].notna().any():
                ax.plot(epochs, df[vlc].to_numpy(), lw=1.4, ls="--",
                        color=_PALETTE[1], label="Validation")
        ax.set_title(title, pad=6)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="best")
    for ax in axes:
        if not ax.get_visible():
            continue
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.tick_params(labelsize=9)
    fig.tight_layout(w_pad=2.2)
    save(fig, out_dir, "fig1_loss_curves")


def _load_protocol(path: str) -> pd.DataFrame | None:
    if not path or not os.path.isfile(path):
        return None
    p = pd.read_csv(path)
    for c in p.columns:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    return p


def fig_metrics(df: pd.DataFrame, out_dir: str, protocol_path: str | None = None) -> None:
    """Figure 2 - metric convergence with annotated peaks.

    Val-set curves come from results.csv; the China-subset protocol result
    (the paper's headline number) is drawn as horizontal reference lines,
    or as a real per-epoch curve when protocol_path is supplied."""
    epochs = _epochs(df)
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    series = [
        (COL_MAP50, "mAP$_{50}$ (val)", _PALETTE[0]),
        (COL_MAP50_95, "mAP$_{50:95}$ (val)", _PALETTE[2]),
        (COL_PREC, "Precision (val)", _PALETTE[3]),
        (COL_RECALL, "Recall (val)", _PALETTE[1]),
    ]
    best = {}
    for col, label, color in series:
        c = _pick(df, col)
        if c is None or df[c].notna().sum() < 2:
            continue
        y = df[c].to_numpy(dtype=float)
        ax.plot(epochs, y, lw=1.5, color=color, label=label, alpha=0.95)
        i = int(np.nanargmax(y))
        best[label] = y[i]
        ax.scatter([epochs[i]], [y[i]], s=22, color=color, edgecolor="white",
                   linewidths=0.6, zorder=4)
        ax.annotate(f"{y[i]:.3f}", (epochs[i], y[i]),
                    xytext=(0, 6), textcoords="offset points", fontsize=8.5,
                    color=color, ha="center")

    x_end = float(epochs.max())
    for pv, label, color, ls in (
        (PROTO_MAP50, "China mAP$_{50}$ (TTA)", _PALETTE[0], "--"),
        (PROTO_MAP50_95, "China mAP$_{50:95}$ (TTA)", _PALETTE[2], "-."),
    ):
        ax.axhline(pv, ls=ls, lw=1.3, color=color, alpha=0.85)
        ax.text(x_end + 0.6, pv, f"{label} {pv:.3f}", fontsize=8.2,
                color=color, ha="left", va="center")
    ax.set_xlim(0, x_end + len(ax.get_xticks()))
    ax.margins(x=0.05)

    proto_peak = {}
    proto = _load_protocol(protocol_path)
    if proto is not None:
        epc = _pick(proto, "epoch")
        for pcol, label, color in (
            ("mAP50", "China mAP$_{50}$ (per-epoch)", _PALETTE[0]),
            ("mAP50-95", "China mAP$_{50:95}$ (per-epoch)", _PALETTE[2]),
        ):
            c = _pick(proto, pcol)
            if c is None or proto[c].notna().sum() < 2:
                continue
            y = proto[c].to_numpy(dtype=float)
            ax.plot(proto[epc].to_numpy(), y, lw=1.5, ls=":", color=color,
                    alpha=0.75, label=label)
            i = int(np.nanargmax(y))
            proto_peak[label] = y[i]
            ax.scatter([proto[epc].to_numpy()[i]], [y[i]], s=26,
                       color=color, edgecolor="white", linewidths=0.6, zorder=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="lower right", ncol=2, borderaxespad=0.4)
    fig.tight_layout()
    print("Val-set peak metrics: "
          + ", ".join(f"{k}={v:.3f}" for k, v in best.items()))
    print("China-protocol (TTA): "
          f"mAP50={PROTO_MAP50:.3f} mAP50-95={PROTO_MAP50_95:.3f} F1={PROTO_F1:.3f}")
    if proto_peak:
        print("China-protocol curve peaks: "
              + ", ".join(f"{k}={v:.3f}" for k, v in proto_peak.items()))
    save(fig, out_dir, "fig2_metric_progression")


def fig_comparison(out_dir: str) -> None:
    """Figure 3 - StripRFNet (baseline) vs RapierNet grouped bars."""
    metrics = ["mAP$_{50}$", "mAP$_{50:95}$", "F1-Score"]
    baseline = np.array([84.4, 52.0, 80.33])
    ours = np.array([83.8, 54.0, 84.1])
    x = np.arange(len(metrics))
    w = 0.32
    fig, ax = plt.subplots(figsize=(7.1, 3.3))
    b1 = ax.bar(x - w / 2, baseline, w, color="#9db4c8", edgecolor="#3c6e9c",
                linewidth=0.8, label="StripRFNet")
    b2 = ax.bar(x + w / 2, ours, w, color="#c0504d", edgecolor="#8f2d2d",
                linewidth=0.8, label="RapierNet (Ours, TTA)")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}",
                        (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left", ncol=2, borderaxespad=0.4)
    ax.xaxis.grid(False)
    fig.tight_layout()
    save(fig, out_dir, "fig3_comparison_bars")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-r", "--results", default="",
                    help="path to Ultralytics results.csv "
                         "(default: auto-search cwd + /kaggle/working)")
    ap.add_argument("-o", "--out", default="plots",
                    help="output directory for figures (default: ./plots)")
    ap.add_argument("-p", "--protocol-csv", default="",
                    help="optional per-epoch China-protocol metrics CSV to "
                         "overlay on fig2 (columns: epoch,mAP50,mAP50-95)")
    args, _ = ap.parse_known_args()

    os.makedirs(args.out, exist_ok=True)
    if not args.results:
        args.results = find_results()
    df = load_results(args.results)
    print(f"Loaded {len(df)} epochs from {args.results}")
    fig_losses(df, args.out)
    fig_metrics(df, args.out, protocol_path=args.protocol_csv)
    fig_comparison(args.out)
    print(f"Done. Figures saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()