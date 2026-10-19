#!/usr/bin/env python3
# Converted from resume-150-epoch(1).ipynb
# RDD2022 training/resume experiment

# %% Cell 1
#!/usr/bin/env python3
# Converted from rdd2022_resume_to_150.ipynb
# RDD2022 resume training: last.pt -> total 150 epochs
# FIXED: proper ultralytics-8.4 resume (same-run continuation from last.pt).
# No fl_gamma (unsupported in this ultralytics), no rebuild-from-yaml (that would
# break resume). We restore the full optimizer/scheduler/EMA/epoch state.

# %% [markdown] Cell 2
# # RDD2022 — Resume Training to 150 Epochs
#
# This notebook resumes the existing Ultralytics training run from `last.pt`. The target is **150 total epochs**, so a checkpoint around epoch 75 should continue for approximately 75 more epochs. The dataset, checkpoint, best checkpoint, and original training arguments are defined in the first code cell.
#
# Do not replace `last.pt` with `best.pt` for resuming: `last.pt` contains the latest training state. `best.pt` is used only for final evaluation if no newer best checkpoint is produced.

# %% Cell 3
import os, sys, gc, math, random, copy, json, time, glob, shutil, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42); torch.manual_seed(42)

print("Python      :", sys.version.split()[0])
print("PyTorch     :", torch.__version__)
print("CUDA avail  :", torch.cuda.is_available())
print("GPU         :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
if torch.cuda.is_available():
    print("GPU mem (GB):", round(torch.cuda.get_device_properties(0).total_memory/1e9, 2))
    torch.backends.cudnn.benchmark = True

# ------------------------------- PATH CONFIG --------------------------------
DATASET = "/kaggle/input/datasets/aliabdelmenam/rdd-2022/RDD_SPLIT"
CHECKPOINT = "/kaggle/input/datasets/amlmoawad/rdd2022-stage2-checkpoints/weights/last.pt"
BEST_CHECKPOINT = "/kaggle/input/datasets/amlmoawad/rdd2022-stage2-checkpoints/weights/best.pt"
ARGS_YAML = "/kaggle/input/datasets/amlmoawad/rdd2022-stage2-checkpoints/args.yaml"
WORKDIR = "/kaggle/working"
TOTAL_EPOCHS = 150                      # overall target (same as run 1)
os.makedirs(WORKDIR, exist_ok=True)

print("Dataset     :", DATASET)
print("Checkpoint  :", CHECKPOINT)
print("Best weight :", BEST_CHECKPOINT)
print("Args YAML   :", ARGS_YAML)
assert os.path.isdir(DATASET), f"Dataset directory not found: {DATASET}"
assert os.path.isfile(CHECKPOINT), f"Checkpoint not found: {CHECKPOINT}"
if os.path.isfile(ARGS_YAML):
    print("   (args.yaml found)")
else:
    ARGS_YAML = None
    print("   (args.yaml NOT found in upload -> will read args from last.pt itself)")

# %% Cell 4
try:
    import ultralytics
    print("ultralytics :", ultralytics.__version__)
    from ultralytics import YOLO
except Exception:
    # On a fresh Kaggle image it is preinstalled; this is a belt-and-suspenders step.
    os.system("pip install -q ultralytics albumentations ensemble-boxes timm -U")
    import ultralytics
    from ultralytics import YOLO

import albumentations as A

try:
    import ensemble_boxes  # Weighted Boxes Fusion
    HAS_WBF = True
except Exception:
    HAS_WBF = False

os.environ.setdefault("WANDB_SILENT", "true")

# %% Cell 5
class DeformableStripConv(nn.Module):
    """Dual-axis strip conv whose offsets are learned => adapts to crack
    orientation instead of a fixed horizontal/vertical kernel grid."""
    def __init__(self, k=7):
        super().__init__()
        self.k = k
        self.offset = None
        self.conv_h = None
        self.conv_v = None
        self.gate = None
        self.act = nn.GELU()

    def _init(self, c, device):
        # lazy weight allocation once the input channel count is known
        if self.conv_h is None or self.conv_h.in_channels != c:
            k, pad = self.k, self.k // 2
            self.conv_h = nn.Conv2d(c, c, (1, k), padding=(0, pad), groups=c,
                                    bias=False, device=device)
            self.conv_v = nn.Conv2d(c, c, (k, 1), padding=(pad, 0), groups=c,
                                    bias=False, device=device)
            self.offset = nn.Conv2d(c, 2, 3, padding=1, bias=True, device=device)
            self.gate = nn.Conv2d(2 * c, c, 1, bias=False, device=device)
            self.act = nn.GELU()

    def forward(self, x):
        c = x.shape[1]
        self._init(c, x.device)
        h, w = x.shape[2], x.shape[3]
        off = self.offset(x).tanh() * 1.5
        b = x.shape[0]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device), indexing="ij")
        grid = torch.stack([xx, yy], -1).unsqueeze(0).expand(b, h, w, 2)
        grid = grid + 2.0 / h * off.permute(0, 2, 3, 1)
        xs = F.grid_sample(x, grid, mode="bilinear", align_corners=False, padding_mode="border")
        hf = self.act(self.conv_h(xs))
        vf = self.act(self.conv_v(xs))
        gf = torch.sigmoid(self.gate(torch.cat([hf, vf], 1)))
        return hf * gf + vf * (1 - gf)

# %% Cell 6
class SARF(nn.Module):
    """Shape-Adaptive Receptive Field block -- channel-preserving drop-in block
    (yaml: `[-1, 1, SARF, []]`). Combines adaptive strip + multi-scale atrous."""
    def __init__(self, k=7, rates=(1, 3, 5)):
        super().__init__()
        self.k = k
        self.rates = rates
        self.strip = None      # lazy DeformableStripConv
        self.asy = None
        self.merge = None
        self.bn = None
        self.act = nn.SiLU()

    def _init(self, c, device):
        r = self.rates
        if self.strip is None or self.strip.gate is None or self.strip.gate.out_channels != c:
            self.strip = DeformableStripConv(self.k)
            c_ = max(16, c // 4)
            self.asy = nn.ModuleList(
                [nn.Conv2d(c, c_, 3, padding=x, dilation=x, device=device) for x in r])
            self.merge = nn.Conv2d(c_ * len(r) + c, c, 1, device=device)
            self.bn = nn.BatchNorm2d(c, device=device)

    def forward(self, x):
        c = x.shape[1]
        self._init(c, x.device)
        s = self.strip(x)
        a = torch.cat([m(x) for m in self.asy], 1)
        out = self.act(self.bn(self.merge(torch.cat([s, a], 1))))
        return out + x

# %% Cell 7
class CoordGate(nn.Module):
    """Spatial+channel attention gate (CBAM-style, boundary-aware).
    Channel-preserving: yaml `[-1, 1, CoordGate, []]`."""
    def __init__(self, r=4):
        super().__init__()
        self.r = r
        self.sp_conv = None
        self.ch_gate = None

    def _init(self, c, device):
        if self.sp_conv is None or self.sp_conv.in_channels != 2:
            c_ = max(8, c // self.r)
            self.sp_conv = nn.Conv2d(2, 1, 7, padding=3, device=device)
            self.ch_gate = nn.Sequential(
                nn.Conv2d(c, c_, 1, device=device), nn.SiLU(),
                nn.Conv2d(c_, c, 1, device=device))

    def forward(self, x):
        c = x.shape[1]
        self._init(c, x.device)
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        sp = torch.sigmoid(self.sp_conv(torch.cat([avg, mx], 1)))
        x = x * sp
        ch = torch.sigmoid(self.ch_gate(x.mean((2, 3), keepdim=True)))
        return x * ch

# %% Cell 8
class DySample(nn.Module):
    """Content-adaptive dynamic upsampling (channel-preserving)."""
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        self.conv = None
        self.offset = None

    def _init(self, c, device):
        if self.conv is None or self.conv.in_channels != c:
            self.conv = nn.Conv2d(c, c, 1, bias=False, device=device)
            self.offset = nn.Conv2d(c, 2, 3, padding=1, device=device)

    def forward(self, x):
        c = x.shape[1]
        self._init(c, x.device)
        x = self.conv(x) + x
        b, _, h, w = x.shape
        s = self.scale
        theta = torch.eye(2, 3, device=x.device).unsqueeze(0).expand(b, -1, -1)
        base = F.affine_grid(theta, (b, 1, h * s, w * s), align_corners=False)
        off = self.offset(x)
        off = F.interpolate(off, size=(h * s, w * s), mode="bilinear",
                            align_corners=False).permute(0, 2, 3, 1)
        grid = base + off * (0.5 / max(h, w))
        out = F.grid_sample(x, grid, mode="bilinear", align_corners=False,
                            padding_mode="border")
        return out

# %% Cell 9
def build_augmentation_pipeline():
    """Advanced Albumentations - road-specific. Note ultralytics runs its own
    mosaic/mixup natively; this extra wrapper adds geometric + color diversity.
    NOTE: On RESUME, albumentations/wrapper augmentation does not matter: the
    wrapper is not applied during training (only used for the optional post-hoc
    inference checks). Training augmentation is fully restored from last.pt."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.12, rotate_limit=20,
                           p=0.6, border_mode=0),
        A.Perspective(scale=(0.02, 0.08), p=0.3),
        A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15,
                                   contrast_limit=0.15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15,
                             val_shift_limit=15, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.25),
        A.CoarseDropout(max_holes=6, max_height=24, max_width=24,
                        fill_value=0, p=0.2),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],
                                min_visibility=0.2))

# %% Cell 10
def check_dataset_layout():
    imgs = glob.glob(f"{DATASET}/**/*.jpg", recursive=True)
    if not imgs:
        imgs = glob.glob(f"{DATASET}/**/*.png", recursive=True)
    print(f"found {len(imgs)} images under {DATASET}")
    assert imgs, f"No images found under {DATASET} - adjust DATASET path."
    dirs = set(os.path.dirname(p) for p in imgs[:200])
    print("sample dirs:", list(dirs)[:6])
    return imgs

# %% Cell 11
def main():
    print("=" * 78)
    print("  RAPIERNET RESUME -> continue last.pt to %d total epochs" % TOTAL_EPOCHS)
    print("=" * 78)

    # ------------------------------- STEP 1 ---------------------------------
    print("\n[Step 1] Checking dataset layout + building augmentation pipeline ...")
    imgs = check_dataset_layout()
    aug = build_augmentation_pipeline()
    print("   Albumentations pipeline ready:", len(aug.transforms), "transforms")

    data_yaml = f"{WORKDIR}/rdd2022.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {DATASET}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("names:\n  0: D00\n  1: D10\n  2: D20\n  3: D40\n")
    print("   data yaml ->", data_yaml)

    # ------------------------------- STEP 2 ---------------------------------
    print("\n[Step 2] Registering custom modules (needed to unpickle last.pt) ...")
    # We DO NOT rebuild the graph here. Resuming must load the exact model from
    # last.pt. Registering the classes is required so torch can unpickle SARF /
    # DySample / CoordGate / DeformableStripConv sheets inside the checkpoint.
    import ultralytics.nn.modules as um
    import ultralytics.nn.tasks as utasks
    _custom = {
        "SARF": SARF,
        "DeformableStripConv": DeformableStripConv,
        "CoordGate": CoordGate,
        "DySample": DySample,
    }
    for _k, _v in _custom.items():
        setattr(um, _k, _v)
        setattr(utasks, _k, _v)
    print("   registered:", list(_custom))

    # ------------------------------- STEP 3 ---------------------------------
    print("\n[Step 3] Resuming training from last.pt ...")

    # --- adaptive batch by available GPU memory (Kaggle: P100/T4/A100 vary) ---
    if torch.cuda.is_available():
        try:
            _mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if _mem_gb >= 30:
                batch = 64
            elif _mem_gb >= 14:
                batch = 32
            else:
                batch = 16
        except Exception:
            batch = 32
    else:
        batch = 8
    print(f"   selected batch={batch} (adapter by VRAM)")

    # ---- Inspect + patch the checkpoint so resume can write under /kaggle/working
    print("\n[Resume] Inspecting checkpoint:", CHECKPOINT)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    ta = ckpt.get("train_args", {})
    start_epoch = ckpt.get("epoch", -1)
    print("   checkpoint epoch (0-indexed):", start_epoch)
    stored_epochs = ta.get("epochs", TOTAL_EPOCHS)
    print("   stored target epochs   :", stored_epochs)
    print("   stored project/name    :", ta.get("project"), "/", ta.get("name"))
    print("   stored save_dir        :", ta.get("save_dir"))
    print("   stored data            :", ta.get("data"))

    # (a) ensure the stored target epoch count is at least TOTAL_EPOCHS so that
    # resume_training's `assert start_epoch < self.epochs` passes for total 150.
    if int(stored_epochs) < TOTAL_EPOCHS:
        ta["epochs"] = TOTAL_EPOCHS
        print(f"   -> bumped stored epochs to {TOTAL_EPOCHS}")
    else:
        ta["epochs"] = TOTAL_EPOCHS

    # (b) repoint data at the current dataset (in case run-1 path differs)
    ta["data"] = data_yaml
    # (c) force a writable save-dir/project/name under /kaggle/working and a
    # clean run id so get_save_dir / get_latest_run behave (ultimate fallback).
    ta["project"] = f"{WORKDIR}/rapiernet_cont"
    ta["name"] = "run150"
    ta.setdefault("resume", str(CHECKPOINT))

    ckpt["train_args"] = ta

    # Where to put the resumable checkpoint so ultralytics writes next to it.
    master_dir = os.path.join(WORKDIR, "rapiernet_cont", "run150")
    wdir = os.path.join(master_dir, "weights")
    os.makedirs(wdir, exist_ok=True)
    patched_pt = os.path.join(wdir, "last.pt")
    # Also keep args.yaml next to it (belt + suspenders for get_latest_run)
    torch.save(ckpt, patched_pt)
    print("   wrote patched resume checkpoint ->", patched_pt)

    # best.pt from run 1 (if any) kept aside for final fallback
    os.makedirs(os.path.join(master_dir, "weights"), exist_ok=True)
    local_best = os.path.join(wdir, "best.pt")
    if os.path.isfile(BEST_CHECKPOINT):
        shutil.copyfile(BEST_CHECKPOINT, local_best)
        BEST_FALLBACK = local_best
        print("   copied run-1 best.pt ->", local_best)
    else:
        BEST_FALLBACK = None

    del ckpt
    gc.collect()

    # ---- Load the patched checkpoint and resume.
    # CRITICAL: with resume=True, ultralytics IGNORES nearly all passed args and
    # re-reads the checkpoint's stored train_args (restoring optimizer, scheduler,
    # EMA, LR schedule, and the exact model). Only these overrides are honored:
    # device, batch, workers, cache, patience, val, plots, imgsz, close_mosaic.
    # So we pass ONLY those. Everything else (lr0, cos_lr, cls, mosaic...) came
    # from run 1 and must stay identical for a TRUE continuation.
    model = YOLO(patched_pt)
    train_args = dict(
        resume=True,
        data=data_yaml,        # data is honored via check_resume when stored data is missing
        device=0 if torch.cuda.is_available() else "cpu",
        batch=batch,
        workers=4,
        cache=False,
    )
    # NOTE on focal loss: this ultralytics build has NO fl_gamma argument (verified),
    # so focal is unavailable. Run 1 was trained with a cls-weight (see args.yaml).
    # We continue with run 1's loss config exactly; passing nothing here keeps it.

    results = model.train(**train_args)

    # Best checkpoint: prefer the newest best.pt written by this resumed run.
    best_pt = os.path.join(wdir, "best.pt")
    if not os.path.isfile(best_pt) or os.path.getsize(best_pt) < 1000:
        best_pt = BEST_FALLBACK
    assert os.path.isfile(best_pt), f"best.pt not found: {best_pt}"
    print("   best weights ->", best_pt)
    if getattr(results, "results_dict", None):
        print("   training summary :",
              {k: round(float(v), 3) for k, v in results.results_dict.items()})

    # ------------------------------- STEP 4 ---------------------------------
    print("\n[Step 4] Inference + confidence tuning + Weighted Boxes Fusion ...")
    best_model = YOLO(best_pt)
    DEV = 0 if torch.cuda.is_available() else "cpu"

    val_dir = DATASET
    for probe in ["images", "test", "val", "train"]:
        p = os.path.join(DATASET, probe)
        if os.path.isdir(p):
            val_dir = p
            break
    holdout = [im for im in glob.glob(f"{val_dir}/**/*.jpg", recursive=True)
               if not any(t in os.path.basename(im) for t in ("_mask", "._"))][:120]
    if not holdout:
        holdout = glob.glob(f"{val_dir}/**/*.png", recursive=True)[:120]

    metrics = best_model.val(data=data_yaml, conf=0.25, iou=0.45,
                             imgsz=640, device=DEV)
    mp, mr = metrics.box.mp, metrics.box.mr
    print("   val mAP50=%.3f  mAP50-95=%.3f  P=%.3f  R=%.3f  F1=%.3f" % (
        metrics.box.map50, metrics.box.map, mp, mr,
        2 * mp * mr / (mp + mr + 1e-9)))

    best_conf = 0.25
    if HAS_WBF and holdout:
        import ensemble_boxes
        print("   WBF: fusing conf=0.10 and conf=0.25 passes ...")
        dets_lo = best_model.predict(source=holdout[:80], conf=0.10, iou=0.5,
                                     imgsz=640, device=DEV, verbose=False)
        dets_hi = best_model.predict(source=holdout[:80], conf=0.25, iou=0.5,
                                     imgsz=640, device=DEV, verbose=False)
        fused_total = 0
        for r_lo, r_hi in zip(dets_lo, dets_hi):
            boxes, scores, labels = [], [], []
            for r in (r_lo, r_hi):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                boxes.append((r.boxes.xyxy / 640.0).cpu().numpy().tolist())
                scores.append(r.boxes.conf.cpu().numpy().tolist())
                labels.append(r.boxes.cls.int().cpu().numpy().tolist())
            if len(boxes) < 2:
                fused = boxes[0] if boxes else []
            else:
                fused = ensemble_boxes.nms_wbf(boxes, scores, labels, iou_thr=0.55, weights=None)
            fused_total += len(fused)
        print("   WBF fused runs over %d images -> %d fused boxes (IoU thr 0.55)"
              % (min(80, len(holdout)), fused_total))
    else:
        print("   (WBF skipped: no ensembling lib / empty holdout - single NMS)")

    if holdout:
        tta = best_model.predict(source=holdout[:40], imgsz=640, conf=best_conf,
                                 iou=0.5, device=DEV, verbose=False, augment=True)
        print("   TTA inference done on sample ->", len(tta), "images")

    if os.environ.get("HF_TOKEN"):
        try:
            best_model.export(format="torchscript")
            best_model.save(f"{WORKDIR}/rapiernet.pt")
            print("   Saved rapiernet.pt  ->  push to HF: rezzzq/rapiernet-rdd2022")
        except Exception as e:
            print("   (HF export skipped:", e, ")")
    else:
        print("   no HF_TOKEN set -> model left local at", best_pt)

    print("\n" + "=" * 78)
    print("  DONE. RAPIERNET resumed to %d total epochs; check val mAP above." % TOTAL_EPOCHS)
    print("=" * 78)

# %% Cell 12
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\n[FATAL] Run aborted. Check traceback above (dataset layout or GPU).")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# %% Cell 13
import os
import shutil

output_dir = "/kaggle/working/rapiernet_cont"
zip_filename = "/kaggle/working/rapiernet_cont.zip"

if os.path.exists(output_dir):
    shutil.make_archive(output_dir, 'zip', output_dir)
    print("Zipped final run ->", zip_filename)
else:
    print("output dir not found, check project path:", output_dir)

# %% Cell 14
# ============================================================
# EVALUATION ONLY — NO TRAINING
# ============================================================
import os
import yaml
import torch
from ultralytics import YOLO

DATASET = "/kaggle/input/datasets/aliabdelmenam/rdd-2022/RDD_SPLIT"
BEST_PT = "/kaggle/working/rapiernet_cont/run150/weights/best.pt"
LAST_PT = "/kaggle/working/rapiernet_cont/run150/weights/last.pt"
EVAL_DIR = "/kaggle/working/evaluation_only"
os.makedirs(EVAL_DIR, exist_ok=True)

assert os.path.isdir(DATASET), f"Dataset not found: {DATASET}"
assert os.path.isfile(BEST_PT), f"best.pt not found: {BEST_PT}"
assert os.path.isfile(LAST_PT), f"last.pt not found: {LAST_PT}"

# Write valid YAML using a YAML serializer.
DATA_YAML = os.path.join(EVAL_DIR, "rdd2022_eval.yaml")
data_config = {
    "path": DATASET,
    "train": "train/images",
    "val": "val/images",
    "names": {
        0: "D00",
        1: "D10",
        2: "D20",
        3: "D40",
    },
}

with open(DATA_YAML, "w", encoding="utf-8") as f:
    yaml.safe_dump(data_config, f, sort_keys=False, allow_unicode=True)

print("Valid YAML written to:", DATA_YAML)
print(open(DATA_YAML, encoding="utf-8").read())

DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Evaluation only — no training will be performed")


def evaluate_checkpoint(weight_path, name):
    print(f"\\n================ {name} ================")
    print("Weights:", weight_path)

    model = YOLO(weight_path)

    metrics = model.val(
        data=DATA_YAML,
        imgsz=640,
        batch=16,
        device=DEVICE,
        workers=2,
        amp=False,
        plots=True,
        project=EVAL_DIR,
        name=name,
        exist_ok=True,
        verbose=True,
    )

    box = metrics.box
    print(f"\\n{name} results:")
    print(f"Precision : {box.mp:.4f}")
    print(f"Recall    : {box.mr:.4f}")
    print(f"mAP50     : {box.map50:.4f}")
    print(f"mAP50-95  : {box.map:.4f}")
    return metrics


best_metrics = evaluate_checkpoint(BEST_PT, "best_eval")
last_metrics = evaluate_checkpoint(LAST_PT, "last_eval")

print("\\nFinished evaluation successfully.")
print("No training was performed.")

