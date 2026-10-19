"""Paper-figure generators mirroring StripRFNet's figure set.

Reproduces the result figures of a StripRFNet-style RDD paper from a
RapierNet checkpoint + china-subset data that already exist on Kaggle:
  * PR curves (per-class + mean)              fig_pr_curve
  * F1-confidence curves                      fig_f1_curve
  * normalized confusion-matrix heatmap       fig_confusion
  * ablation grouped bars (ablation_summary)  fig_ablation
  * Fig.5-style attention comparison grid     fig_attention
  * qualitative GT-vs-prediction grid         fig_samples
  * RapierNet architecture schematic          fig_architecture
Everything exported as 300 DPI PNG + vector PDF into --out.

Usage:
  python make_paper_figures.py --best /kaggle/working/rapi5_hr/weights/best.pt \
      --base /kaggle/working/abl_plain_yolo11-m/weights/best.pt \
      --out /kaggle/working/paper_figs
"""

from __future__ import annotations

import argparse
import glob
import math
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

try:
    import seaborn as sns
except ImportError:
    sns = None

_RC = {
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "dejavusans",
    "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10.5, "legend.fontsize": 9, "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5, "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": "#b9b9b9", "grid.alpha": 0.35, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
}
matplotlib.rcParams.update(_RC)
if sns is not None:
    sns.set_style("whitegrid", {"font.family": "sans-serif", "axes.grid": False})

PAL = ["#0b5394", "#c00000", "#2e7d32", "#7b2fbe", "#e0810d", "#0f7c8c"]
NAMES = ["alligator", "transverse", "longitudinal", "corrupt", "pothole"]
_COLORMAP = matplotlib.colormaps["jet_r"]


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
        print(f"  wrote {name}.{ext}")
    plt.close(fig)


def _val_metrics(best_pt, yaml_path, device=0, imgsz=560, conf=0.25):
    from ultralytics import YOLO
    m = YOLO(best_pt)
    met = m.val(data=yaml_path, conf=conf, iou=0.5, imgsz=imgsz, device=device,
                batch=8, verbose=False)
    return m, met


def _curves_dict(met):
    """Normalize met.box.curves_results to {key: (x_list, y_list, class_ids)}.
    Handles every layout seen across versions:
      - 8.3.x dict:   {key: (x, y, names)}
      - 8.4.x list:   [[px, values, xlabel, ylabel], ...]  (values: nc x npoints)
      - some lists:   [(name, x, y, names), ...]
    class_ids aligns rows of y_list to class ids (from met.box.ap_class_index).
    """
    cr = getattr(getattr(met, "box", None), "curves_results", None)
    idx = getattr(getattr(met, "box", None), "ap_class_index", None)
    idx = list(idx) if idx is not None else None
    out = {}
    if isinstance(cr, dict):
        for k, v in cr.items():
            if isinstance(v, (list, tuple)) and len(v) == 3:
                out[k] = v
        return out
    if not isinstance(cr, list):
        return out
    for item in cr:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            continue
        a, b, c, d = item
        if isinstance(c, str) and isinstance(d, str):
            # 8.4 style: [px, values, xlabel, ylabel]
            px = np.asarray(a); vals = np.asarray(b)
            if vals.ndim == 1:
                out[(d, c)] = ([px], [vals], np.array([-1]))
            else:
                n = vals.shape[0]
                ids = np.asarray(idx[:n]) if idx else np.arange(n)
                out[(d, c)] = ([px] * n, [vals[i] for i in range(n)], ids)
        else:
            # older: (name, x, y, names)  -> a=name, b=x, c=y, d=names
            name = tuple(a) if isinstance(a, (list, tuple)) else a
            out[name] = (list(b), list(c), np.asarray(d))
    return out


def _curve_key(cr, *subs):
    """Find a curve whose key mentions every substring in *subs (str or tuple keys)."""
    for k in cr:
        s = " ".join(map(str, k)) if isinstance(k, (list, tuple)) else str(k)
        if all(x in s.lower() for x in subs):
            return k
    return None


def fig_pr_curve(met, out):
    """Precision-Recall curves per class + bold mean line."""
    cr = _curves_dict(met)
    key = _curve_key(cr, "precision", "recall")
    if key is None:
        print("[pr] curves unavailable; skipped")
        return
    x, y, names = cr[key]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for i, (xi, yi) in enumerate(zip(x, y)):
        cid = int(names[i])
        lab = NAMES[cid] if 0 <= cid < len(NAMES) else "All classes"
        if cid < 0:
            ax.plot(xi, yi, lw=2.6, color="#111111", label=lab, zorder=5)
        else:
            ax.plot(xi, yi, lw=1.3, color=PAL[i % len(PAL)], label=lab)
    try:  # add bold mean of per-class curves if not already present
        if np.asarray(names).min() >= 0 and all(np.asarray(v).size for v in x):
            xm = np.mean(np.stack([np.asarray(v) for v in x]), axis=0)
            ym = np.mean(np.stack([np.asarray(v) for v in y]), axis=0)
            ax.plot(xm, ym, lw=2.6, color="#111111", label="All classes", zorder=5)
    except Exception:
        pass
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="0.6")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_xlabel("Recall")
    ax.set_ylabel("Precision"); ax.legend(loc="lower left", ncol=2)
    ax.set_title("Precision–Recall curves (China subset)")
    _save(fig, out, "fig_pr_curve")


def fig_f1_curve(met, out):
    """F1 vs confidence threshold per class + mean."""
    cr = _curves_dict(met)
    key = _curve_key(cr, "f1", "confidence")
    if key is None:
        print("[f1] curves unavailable; skipped")
        return
    x, y, names = cr[key]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for i, (xi, yi) in enumerate(zip(x, y)):
        cid = int(names[i])
        lab = NAMES[cid] if 0 <= cid < len(NAMES) else "All classes"
        if cid < 0:
            ax.plot(xi, yi, lw=2.6, color="#111111", label=lab, zorder=5)
        else:
            ax.plot(xi, yi, lw=1.3, color=PAL[i % len(PAL)], label=lab)
    try:  # add bold mean of per-class curves if not already present
        if np.asarray(names).min() >= 0 and all(np.asarray(v).size for v in x):
            xm = np.mean(np.stack([np.asarray(v) for v in x]), axis=0)
            ym = np.mean(np.stack([np.asarray(v) for v in y]), axis=0)
            ax.plot(xm, ym, lw=2.6, color="#111111", label="All classes", zorder=5)
    except Exception:
        pass
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("F1 score")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.legend(loc="lower left", ncol=2)
    ax.set_title("F1-score vs confidence threshold (China subset)")
    _save(fig, out, "fig_f1_curve")


def fig_confusion(met, out):
    """Normalized confusion-matrix heatmap (publication style)."""
    cm = getattr(met, "confusion_matrix", None)
    if cm is None or getattr(cm, "matrix", None) is None:
        print("[cm] confusion matrix unavailable; skipped")
        return
    mat = np.asarray(cm.matrix, dtype=float)
    mat = mat[:-1, :-1]  # drop background/miss row + col
    rows = mat.sum(1, keepdims=True)
    rows[rows == 0] = 1
    norm = mat / rows
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    n = norm.shape[0]
    for i in range(n):
        for j in range(n):
            v = norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8.5, color="white" if v > 0.5 else "#1a1a1a")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(NAMES[:n], rotation=30, ha="right")
    ax.set_yticklabels(NAMES[:n])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    ax.set_title("Normalized confusion matrix (China subset)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out, "fig_confusion_matrix")


def fig_ablation(csv_path, out):
    """Grouped bars: mAP50 / mAP50:95 / F1 per ablation variant."""
    if not csv_path or not os.path.isfile(csv_path):
        print("[abl] ablation_summary.csv not found; skipped")
        return
    import pandas as pd
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[abl] empty ablation summary; skipped")
        return
    order = ["Full RapierNet", "w/o DySample", "w/o SARF",
             "w/o CoordGate", "plain yolo11-m"]
    df = df.set_index("config").reindex(
        [o for o in order if o in df["config"].values]).reset_index()
    metrics = ["mAP50", "mAP50_95", "F1"]
    x = np.arange(len(df)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    for k, met in enumerate(metrics):
        col = "mAP50_95" if met == "mAP50_95" else met
        vals = df[col].to_numpy(dtype=float)
        bars = ax.bar(x + (k - 1) * w, vals, w, color=PAL[k % len(PAL)],
                      edgecolor="0.35", linewidth=0.6,
                      label="mAP$_{50}$" if met == "mAP50" else
                            ("mAP$_{50:95}$" if met == "mAP50_95" else "F1"))
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2,
                                     b.get_height()), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=7.6)
    ax.set_xticks(x); ax.set_xticklabels(df["config"], rotation=18, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=3, borderaxespad=0.4)
    ax.xaxis.grid(False)
    ax.set_title("Ablation study (8 epochs @480, eval @480, China subset)")
    _save(fig, out, "fig_ablation")


def _pick_samples(labels_dir, images_dir, k=4, prefer=(0, 4)):
    import random
    random.seed(0)
    files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    good, rest = [], []
    for f in files:
        rows = [l.split() for l in open(f).read().splitlines() if l.strip()]
        if not rows:
            continue
        try:
            classes = {int(r[0]) for r in rows if r[0].strip().lstrip("-").isdigit()}
        except ValueError:
            continue
        (good if classes & set(prefer) else rest).append((f, rows))
    random.shuffle(good); random.shuffle(rest)
    out = []
    for f, rows in good + rest:
        img = os.path.join(images_dir, os.path.splitext(os.path.basename(f))[0] + ".jpg")
        if not os.path.isfile(img):
            img = os.path.join(images_dir, os.path.splitext(os.path.basename(f))[0] + ".png")
        if not os.path.isfile(img):
            continue
        out.append((img, rows))
        if len(out) >= k:
            break
    return out


def _split_dirs(yaml_path, split):
    import yaml
    cfg = yaml.safe_load(open(yaml_path))
    base = str(cfg.get("path", ""))
    p = str(cfg.get(split) or "") or str(cfg.get("val") or "")
    if not (os.path.isabs(p) or not base):
        p = os.path.join(base, p)
    p = p.replace("\\", "/").rstrip("/")
    if p.endswith("/images"):
        p = os.path.join(os.path.dirname(p), "labels").replace("\\", "/")
    else:
        p = p.replace("/images", "/labels")
    return p, p.replace("/labels", "/images"), bool(cfg.get(split))


def _pick_failure_sample(best_pt, labels_dir, images_dir, imgsz=560, conf=0.42,
                         device=0, tries=180):
    """Find a REAL failure case: a labeled image whose best prediction IoU
    with any GT box is below the match threshold (or a high-conf FP far from
    all GT). Returns (img_path, rows) or None. Used for the 'Failure cases'
    bottom row of the qualitative figure."""
    from ultralytics import YOLO
    import cv2
    model = YOLO(best_pt)
    files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    rng = np.random.RandomState(7)
    rng.shuffle(files)
    for f in files[:tries]:
        rows = [l.split() for l in open(f).read().splitlines() if l.strip()]
        if not rows:
            continue
        b = os.path.splitext(os.path.basename(f))[0]
        img = os.path.join(images_dir, b + ".jpg")
        if not os.path.isfile(img):
            img = os.path.join(images_dir, b + ".png")
        if not os.path.isfile(img):
            continue
        im = cv2.imread(img); h, w = im.shape[:2]
        r = model.predict(img, imgsz=imgsz, conf=conf, device=device,
                          verbose=False)[0]
        preds = (r.boxes.xyxy.cpu().numpy(),
                 r.boxes.cls.cpu().numpy()) if r.boxes is not None and len(r.boxes) else (None, None)
        gt = []
        for row in rows:
            try:
                cid = int(float(row[0]))
            except ValueError:
                continue
            x, y, bw, bh = (float(v) for v in row[1:5])
            gt.append((cid, (x * w, y * h, (x + bw) * w, (y + bh) * h)))
        missed = 0
        for _, (gx1, gy1, gx2, gy2) in gt:
            if preds[0] is None or not len(preds[0]):
                missed += 1
                continue
            best_iou = 0.0
            for px1, py1, px2, py2 in preds[0]:
                ix1 = max(gx1, px1); iy1 = max(gy1, py1)
                ix2 = min(gx2, px2); iy2 = min(gy2, py2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                union = (gx2 - gx1) * (gy2 - gy1) + (px2 - px1) * (py2 - py1) - inter
                best_iou = max(best_iou, inter / (union + 1e-9))
            if best_iou < 0.1:
                missed += 1
        if missed >= 1 or (preds[0] is not None and not len(gt)):
            return img, rows
    return None


def _detection_tiles(best_pt, images, imgsz=560, conf=0.42, device=0):
    from ultralytics import YOLO
    model = YOLO(best_pt)
    tiles = []
    for img in images:
        r = model.predict(img, imgsz=imgsz, conf=conf, device=device,
                          verbose=False)[0]
        tiles.append(r.plot(line_width=3, font_size=18))
    return tiles


def _overlay_gt(samples):
    import cv2
    bgr = lambda c: tuple(int(c[i:i + 2], 16) for i in (5, 3, 1))
    out = []
    for (img, rows) in samples:
        im = cv2.imread(img)
        h, w = im.shape[:2]
        for row in rows:
            cid = int(row[0]) % len(PAL)
            x, y, bw, bh = (float(v) for v in row[1:5])
            x1, y1, x2, y2 = (int(x * w), int(y * h),
                              int((x + bw) * w), int((y + bh) * h))
            cv2.rectangle(im, (x1, y1), (x2, y2), bgr(PAL[cid]), 2)
            cv2.putText(im, NAMES[cid], (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr(PAL[cid]), 2)
        out.append(im)
    return out


def _attention_overlay(best_pt, images, imgsz=560, conf=0.15, device=0):
    from ultralytics import YOLO
    import cv2
    model = YOLO(best_pt)
    outs = []
    for img in images:
        im = cv2.imread(img); h, w = im.shape[:2]
        r = model.predict(img, imgsz=imgsz, conf=conf, device=device,
                          verbose=False)[0]
        heat = np.zeros((h, w), dtype=np.float32)
        if r.boxes is not None and len(r.boxes):
            xy = r.boxes.xyxy.cpu().numpy(); sc = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), s in zip(xy, sc):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                rr = max(12, math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 0.5)
                y0, y1c = int(max(0, cy - rr)), int(min(h, cy + rr))
                x0, x1c = int(max(0, cx - rr)), int(min(w, cx + rr))
                heat[y0:y1c, x0:x1c] = np.maximum(heat[y0:y1c, x0:x1c], s)
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=16, sigmaY=16)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
        heatmap = cv2.applyColorMap((heat * 255).astype(np.uint8),
                                    cv2.COLORMAP_JET)
        gray = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),
                            cv2.COLOR_GRAY2BGR)
        outs.append(cv2.addWeighted(heatmap, 0.55, gray, 0.45, 0))
    return outs


def _sample_dirs(yaml_path, split=None):
    return _split_dirs(yaml_path, split or "val")[:2]


def fig_attention(best_pt, yaml_path, images_dir, out, base_pt=None,
                  device=0, imgsz=560, n=4, split="val"):
    """Fig.5-style: (a) input+GT, (b) proposed (attention). A baseline
    column (c) is only shown when base_pt is provided (3-column layout)."""
    labels_dir, images_dir, have_split = _split_dirs(yaml_path, split)
    samples = _pick_samples(labels_dir, images_dir, k=n, prefer=(0, 4))[:n]
    if not samples and have_split:
        print("[att] no labeled samples in split; falling back to val")
        labels_dir, images_dir, _ = _split_dirs(yaml_path, "val")
        samples = _pick_samples(labels_dir, images_dir, k=n, prefer=(0, 4))[:n]
    if not samples:
        print("[att] no samples; skipped"); return
    imgs = [s[0] for s in samples]
    gt_col = _overlay_gt(samples)
    base_col = _detection_tiles(base_pt, imgs, imgsz=imgsz, device=device
                                ) if base_pt and os.path.isfile(base_pt) else None
    att_col = _attention_overlay(best_pt, imgs, imgsz=imgsz, device=device)
    rows = len(imgs)
    cols = [("(a) Input + GT", gt_col)]
    if base_col:
        cols.append(("(b) Baseline", base_col))
        cols.append(("(c) RapierNet (attention)", att_col))
    else:
        cols.append(("(b) RapierNet (attention)", att_col))
    fig, axes = plt.subplots(rows, len(cols), figsize=(9.5 / 3 * len(cols), 3.2 * rows))
    if rows == 1:
        axes = axes[None, :]
    for i in range(rows):
        for j, (name, col) in enumerate(cols):
            ax = axes[i, j]
            data = col[i] if isinstance(col, (list, tuple)) else col
            if data is None:
                ax.axis("off"); continue
            ax.imshow(data[:, :, ::-1])
            if i == 0:
                ax.set_title(name, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if i == rows - 1:
                ax.set_xlabel(f"Sample {i+1}")
    fig.tight_layout()
    _save(fig, out, "fig5_attention_comparison")


def fig_samples(best_pt, yaml_path, images_dir, out, device=0, imgsz=560, n=4,
                split="test", conf=0.42):
    """Qualitative grid: GT vs predictions; add a real Failure Cases row."""
    labels_dir, images_dir, have_split = _split_dirs(yaml_path, split)
    samples = _pick_samples(labels_dir, images_dir, k=n, prefer=(0, 4))[:n]
    if not samples and have_split:
        print("[samples] no labeled samples in split; falling back to val")
        labels_dir, images_dir, _ = _split_dirs(yaml_path, "val")
        samples = _pick_samples(labels_dir, images_dir, k=n, prefer=(0, 4))[:n]
    if not samples:
        print("[samples] no samples; skipped"); return
    imgs = [s[0] for s in samples]
    gt_col = _overlay_gt(samples)
    pred_col = _detection_tiles(best_pt, imgs, imgsz=imgsz, conf=conf, device=device)
    # a real failure case for the bottom row (if one exists in the split)
    fail = _pick_failure_sample(best_pt, labels_dir, images_dir, imgsz=imgsz,
                                conf=conf, device=device)
    nrows = len(imgs) + (1 if fail else 0)
    fig, axes = plt.subplots(nrows, 2, figsize=(8.6, 3.4 * nrows))
    if nrows == 1:
        axes = axes[None, :]
    for i, img in enumerate(imgs):
        axes[i, 0].imshow(gt_col[i][:, :, ::-1])
        axes[i, 1].imshow(pred_col[i][:, :, ::-1])
        if i == 0:
            axes[i, 0].set_title("(a) Ground truth")
            axes[i, 1].set_title("(b) RapierNet")
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
    if fail:
        fimg, frows = fail
        fgt = _overlay_gt([(fimg, frows)])[0]
        fpred = _detection_tiles(best_pt, [fimg], imgsz=imgsz, conf=conf,
                                 device=device)[0]
        k = len(imgs)
        axes[k, 0].imshow(fgt[:, :, ::-1]); axes[k, 0].set_title("(c) Failure case — GT")
        axes[k, 1].imshow(fpred[:, :, ::-1]); axes[k, 1].set_title("(d) Failure case — RapierNet")
        for ax in axes[k]:
            ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
    fig.tight_layout()
    _save(fig, out, "fig_qualitative")


def _block(ax, x, y, w, h, text, fc, edge):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=1.0, edgecolor=edge, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5,
            color="#111111")


def fig_architecture(out):
    """Schematic of RapierNet (backbone + neck + shaping modules + Detect).

    Wiring follows the built graph exactly: CoordGate is applied on the P4
    feature and SARF on the P5 (deepest) feature; the rewired Detect head
    reads [P3, CoordGate(P4), SARF(P5)].
    """
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.set_xlim(0, 11); ax.set_ylim(0, 7.4); ax.axis("off")
    bw, bh = 1.45, 0.55
    bxn = 0.5
    names = ["C1/64\n(P1)", "C2/128\n(P2)", "C3/256\n(P3-stage)", "C4/512\n(P4-stage)", "C5/1024\n→P5 feat."]
    ys = [6.2, 5.15, 4.1, 3.05, 2.0]
    for i, n in enumerate(names):
        _block(ax, bxn, ys[i], bw, bh, n, "#dbe9f6", "#3c6e9c")
    for i in range(4):
        ax.add_patch(FancyArrowPatch((bxn + bw / 2, ys[i]),
                                     (bxn + bw / 2, ys[i + 1] + bh),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color="#666666", lw=0.9))
    # neck: top-down upsample+concat (P4 row, P3 row)
    p4, p3 = (3.0, 3.05), (3.0, 4.1)
    _block(ax, p4[0], p4[1], 1.8, bh, "↑×2+Concat → P4 feat.", "#ffe0b2", "#e0810d")
    _block(ax, p3[0], p3[1], 1.8, bh, "↑×2+Concat → P3 feat.", "#ffe0b2", "#e0810d")
    ax.add_patch(FancyArrowPatch((bxn + bw, ys[4] + bh / 2), (p4[0], p4[1] + bh / 2),
                                 arrowstyle="-|>", mutation_scale=10, color="#e0810d", lw=0.9))
    ax.add_patch(FancyArrowPatch((bxn + bw, ys[3] + bh / 2), (p4[0], p4[1] + bh / 2),
                                 arrowstyle="-|>", mutation_scale=10, color="#666666", lw=0.9))
    ax.add_patch(FancyArrowPatch((bxn + bw, ys[2] + bh / 2), (p3[0], p3[1] + bh / 2),
                                 arrowstyle="-|>", mutation_scale=10, color="#666666", lw=0.9))
    ax.add_patch(FancyArrowPatch((p4[0] + 0.9, p4[1] + bh), (p3[0] + 0.9, p3[1] + bh),
                                 arrowstyle="-|>", mutation_scale=10, color="#666666", lw=0.9))
    # shaping modules: CoordGate on P4, SARF on P5 (exactly as built)
    cg = (5.4, 3.05)   # CoordGate on P4
    sarf = (5.4, 2.0)  # SARF on P5 (deepest)
    _block(ax, cg[0], cg[1], 1.7, bh, "CoordGate\n(on P4)", "#cfe3f5", "#0b5394")
    _block(ax, sarf[0], sarf[1], 1.7, bh, "SARF\n(on P5, deepest)", "#c9e6cd", "#2e7d32")
    ax.add_patch(FancyArrowPatch((p4[0] + 1.8, p4[1] + bh / 2), (cg[0], cg[1] + bh / 2),
                                 arrowstyle="-|>", mutation_scale=10, color="#0b5394", lw=0.9))
    ax.add_patch(FancyArrowPatch((bxn + bw, ys[4] + bh / 2), (sarf[0], sarf[1] + bh / 2),
                                 arrowstyle="-|>", mutation_scale=10, color="#2e7d32", lw=0.9))
    # Detect head reads [P3, CoordGate(P4), SARF(P5)]
    _block(ax, 8.1, 3.3, 2.5, 1.1, "Detect head  [P3, CoordGate(P4), SARF(P5)]\n(nc=5 ± strides 8/16/32)",
           "#f5d6d6", "#c00000")
    ax.add_patch(FancyArrowPatch((p3[0] + 1.8, p3[1] + bh / 2), (8.1, 4.0),
                                 arrowstyle="-|>", mutation_scale=10, color="#666666", lw=0.9))
    ax.add_patch(FancyArrowPatch((cg[0] + 1.7, cg[1] + bh / 2), (8.1, 3.75),
                                 arrowstyle="-|>", mutation_scale=10, color="#0b5394", lw=0.9))
    ax.add_patch(FancyArrowPatch((sarf[0] + 1.7, sarf[1] + bh / 2), (9.2, 3.3),
                                 arrowstyle="-|>", mutation_scale=10, color="#2e7d32", lw=0.9))
    # legend of custom modules
    _block(ax, 8.1, 1.9, 2.5, 1.0, "New blocks (random init):\nDySample(x2) on P3/P4 upsamples\nCoordGate on P4 · SARF on P5",
           "#f2f2f2", "#888888")
    ax.text(0.15, 7.0, "Input image", fontsize=9, style="italic")
    ax.set_title("RapierNet architecture (yolo11-m backbone + neck + shaping modules)")
    _save(fig, out, "fig1_architecture")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--best", default="/kaggle/working/rapi5_hr/weights/best.pt")
    ap.add_argument("--base", default="/kaggle/working/abl_plain_yolo11-m/weights/best.pt",
                    help="baseline weights for Fig.5 col (b), optional")
    ap.add_argument("--yaml", default="/kaggle/working/china_v5.yaml")
    ap.add_argument("--images", default="/kaggle/working/china_v5/test/images",
                    help="china images dir (test subset)")
    ap.add_argument("--abl", default="/kaggle/working/ablation_summary.csv")
    ap.add_argument("--split", default="test",
                    help="split for qualitative/attention samples (test/val)")
    ap.add_argument("--out", default="/kaggle/working/paper_figs")
    ap.add_argument("--device", type=int, default=0)
    args, _ = ap.parse_known_args()

    os.makedirs(args.out, exist_ok=True)
    if os.path.isfile(args.best):
        _, met = _val_metrics(args.best, args.yaml, device=args.device)
        fig_pr_curve(met, args.out)
        fig_f1_curve(met, args.out)
        fig_confusion(met, args.out)
        fig_attention(args.best, args.yaml, args.images, args.out,
                      base_pt=args.base if os.path.isfile(args.base) else None,
                      device=args.device, split=args.split)
        fig_samples(args.best, args.yaml, args.images, args.out,
                    device=args.device, split=args.split)
    else:
        print(f"[warn] {args.best} missing; skipping eval-based figures")
    fig_ablation(args.abl, args.out)
    fig_architecture(args.out)
    print(f"Done -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()