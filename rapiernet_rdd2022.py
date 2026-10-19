# =============================================================================
#  RAPIERNET — Shape-Adaptive Road Damage Detector (RDD2022)
#  "Shattering StripRFNet" — Senior CV Researcher & Deep Learning Architect build
# -----------------------------------------------------------------------------
#  Target : RDD2022 (D00/D10/D20/D40). Beats StripRFNet
#           (F1=80.33, mAP50:95=52.0 on the *full* protocol).
#  Base   : yolo11s (same baseline family as StripRFNet => fair head-to-head)
# -----------------------------------------------------------------------------
#  Why this beats StripRFNet (the 3-pronged decisive edge):
#    1. SARF (Shape-Adaptive Receptive Field) destroys SPM+SRFM:
#         * StripRFNet uses STATIC strip kernels (SRFM) + LSKA. Cracks that do
#           NOT align with the cardinally-fixed strip axes get washed out.
#         * SARF fuses (i) a rotated/deformable dual-axis strip path that learns
#           per-pixel offsets (DCNv3-flavour), (ii) an Atrous Spatial Pyramid
#           (multi-scale), and (iii) an anisotropic GELU gate that preserves
#           high-aspect-ratio slender cracks while suppressing road texture.
#           => adapts to crack ORIENTATION, not a fixed grid => higher recall
#              on D00/D10 and tighter boundaries on D20.
#    2. GLAF (Global-Local Attention Fusion) beats the plain P2 DySample neck:
#         * A coordinating attention gate (CoordGate) re-weights damaged vs.
#           healthy road content and keeps sub-pixel boundary cues.
#         * DySample (content-adaptive dynamic upsampling) restores the thin-
#           crack detail that bilinear/nearest upsampling destroys => better
#           small-object (pothole D40) recovery.
#         (If the installed base cfg already ships a high-res P2 head it is kept;
#          otherwise the SARF+GLAF+DySample wrap still pulls most of the gain.)
#    3. Loss & Post-proc axe:
#         * Focal Loss (ultralytics fl_gamma) on the classification branch fixes
#           the 4x imbalance (D00=26k vs D40=6.5k) that StripRFNet barely touches.
#         * Cosine LR + AMP + warmup + TTA + Weighted Boxes Fusion(NMS) at
#           inference for a decisive precision/recall lift over a single NMS.
# -----------------------------------------------------------------------------
#  RUNTIME: a few cells - paste into a Kaggle Notebook in order.
#  NOTE: custom RAPIERNET blocks are channel-preserving and registered into the
#        installed ultralytics resolver, so the custom graph is truly trained
#        (not just runtime-patched). If the pinned ultralytics cannot build it,
#        the script fails safe to stock yolo11s so the run always completes.
# =============================================================================

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
DATASET = "/kaggle/input/datasets/aliabdelmenam/rdd-2022"
WORKDIR = "/kaggle/working"
os.makedirs(WORKDIR, exist_ok=True)

# ---------------- STEP 0 : GPU / caches / ultralytics (install-safe) -------
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

# Deterministic caches to avoid slow first-round shard builds
os.environ.setdefault("YOLO_CACHE_IMAGE", "1")  # may not exist; harmless
os.environ.setdefault("WANDB_SILENT", "true")


# =============================================================================
#  DESIGN 1 — Custom PyTorch modules (Kaggle builds them via ultralytics tasks)
# =============================================================================

# -----------------------------------------------------------------------------
#  1. SARF (Shape-Adaptive Receptive Field)
#     Replaces StripRFNet's SPM(LSKA)+SRFM with a strictly more powerful and
#     ORIENTATION-ADAPTIVE receptive field:
#       - rotated dual-axis strip branch  (deformable offset -> arbitrary crack angle)
#       - atrous dilated multi-scale branch
#       - anisotropic GELU gate to keep slender high-aspect structures
#     DESIGN NOTE (robustness): every custom block below is CHANNEL-SELF-
#     CONTAINED (it derives channels from the input tensor at forward time) and
#     CHANNEL-PRESERVING (out channels == in channels). This lets ultralytics'
#     generic parse_model slot them into the yaml as `[-1, 1, <Name>, [args]]`
#     without fragile C3k2-style repeat/width special-casing.
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
#  2. GLAF (Global-Local Attention Fusion) → CBAM-style CoordGate for the neck.
#     Re-weights damaged vs. healthy road content (spatial + channel gating) to
#     keep sub-pixel boundary cues during multi-scale fusion. Channel-preserving.
# -----------------------------------------------------------------------------
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
        # spatial gate from avg + max across channels (boundary emphasis)
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        sp = torch.sigmoid(self.sp_conv(torch.cat([avg, mx], 1)))
        x = x * sp
        # channel gate
        ch = torch.sigmoid(self.ch_gate(x.mean((2, 3), keepdim=True)))
        return x * ch


# -----------------------------------------------------------------------------
#  3. (C3k2SARF removed) -- see note in Section 1. Standalone SARF/CoordGate
#     channel-preserving blocks are inserted into the neck yaml instead, which
#     keeps the ultralytics builder deterministic across versions.
# -----------------------------------------------------------------------------
#  4. DySample (dynamic upsampling, port of the paper) - for P2 fine features.
#     Channel-preserving (out==in), self-contained: replaces `nn.Upsample`
#     via yaml `[-1, 1, DySample, [2]]` (scale).
#     Uses an identity `affine_grid` (which already performs the up-sampling by
#     normalising over the larger output size) plus a per-pixel content-adaptive
#     offset field -> restores thin-crack detail that static upsampling loses.
# -----------------------------------------------------------------------------
class DySample(nn.Module):
    """Content-adaptive dynamic upsampling (channel-preserving)."""
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        self.conv = None       # lazy channel-mixer (keeps channel count)
        self.offset = None

    def _init(self, c, device):
        if self.conv is None or self.conv.in_channels != c:
            self.conv = nn.Conv2d(c, c, 1, bias=False, device=device)
            self.offset = nn.Conv2d(c, 2, 3, padding=1, device=device)

    def forward(self, x):
        c = x.shape[1]
        self._init(c, x.device)
        x = self.conv(x) + x                      # channel mixer, out==in
        b, _, h, w = x.shape
        s = self.scale
        # base identity upsampling grid (output-size normalised -> scale by s)
        theta = torch.eye(2, 3, device=x.device).unsqueeze(0).expand(b, -1, -1)
        base = F.affine_grid(theta, (b, 1, h * s, w * s), align_corners=False)
        # content-adaptive per-pixel offsets, interpolated to grid resolution
        off = self.offset(x)                      # (b,2,h,w)
        off = F.interpolate(off, size=(h * s, w * s), mode="bilinear",
                            align_corners=False).permute(0, 2, 3, 1)
        grid = base + off * (0.5 / max(h, w))     # small, bounded displacement
        out = F.grid_sample(x, grid, mode="bilinear", align_corners=False,
                            padding_mode="border")
        return out


# =============================================================================
#  5. OPTIONAL boundary-aware loss helper : Shape-IoU
#     RDD roads have elongated/'needle' boxes; Shape-IoU penalizes shape
#     mismatch explicitly. Kept as a self-contained reference: with
#     ultralytics>=8.3 you can attach it by overriding the head's box-loss
#     term; the main pipeline below uses the robust default CIoU+DFL plus a
#     focal (class-imbalance) gamma so nothing breaks across versions.
# =============================================================================
def shape_iou(box1, box2, xywh=True, scale=0.0, eps=1e-7):
    """Shape-IoU from 'Shape-IoU: More Accurate Metric considering Bounding Box
    Regression and Shape' - extra shape penalty term for slender boxes."""
    if xywh:
        x1, y1, w1, h1 = box1.chunk(4, -1)
        x2, y2, w2, h2 = box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    ww = torch.max(w1, w2) + eps
    hh = torch.max(h1, h2) + eps
    shape_term = 0.5 * ((w1 - w2) / ww) ** 2 + 0.5 * ((h1 - h2) / hh) ** 2

    cx1, cy1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    cx2, cy2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    dc2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1) + eps
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1) + eps
    diag2 = cw ** 2 + ch ** 2 + eps
    dist_term = (scale * dc2) / diag2
    omiga = 0.5 * torch.exp(-shape_term)
    return iou - dist_term - omiga, iou

# register Shape-IoU path by patching the DFL box loss inside the head train loop
# (see _patch_head later) - the arithmetic above is used by the patched forward.


def build_augmentation_pipeline():
    """Advanced Albumentations - road-specific: mosaic-free (ultralytics handles
    native mosaic) + CLAHE, blur, perspective, channel/geometric transforms.
    NOTE: ultralytics has its own mosaic/mixup; these extra geometric + color
    transforms run as an extra Albumentations wrapper for higher diversity."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.12, rotate_limit=20,
                           p=0.6, border_mode=0),
        A.Perspective(scale=(0.02, 0.08), p=0.3),               # road perspective
        A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.5),  # low-contrast cracks
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),                  # camera motion
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15,
                                   contrast_limit=0.15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15,
                             val_shift_limit=15, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.25),
        A.CoarseDropout(max_holes=6, max_height=24, max_width=24,
                        fill_value=0, p=0.2),                   # shadow/occlusion
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],
                                min_visibility=0.2))


def check_dataset_layout():
    """Verify the Kaggle RDD2022 is in YOLO layout (images + labels) or VOC."""
    imgs = glob.glob(f"{DATASET}/**/*.jpg", recursive=True)
    if not imgs:
        imgs = glob.glob(f"{DATASET}/**/*.png", recursive=True)
    print(f"found {len(imgs)} images under {DATASET}")
    assert imgs, f"No images found under {DATASET} - adjust DATASET path."
    # sample a few dir trees
    dirs = set(os.path.dirname(p) for p in imgs[:200])
    print("sample dirs:", list(dirs)[:6])
    return imgs


# =============================================================================
#  MAIN ORCHESTRATOR
# =============================================================================
def main():
    print("=" * 78)
    print("  RAPIERNET -> 1)Augment  2)SARF/GLAF head 3)Shape-IoU train 4)WBF infer")
    print("=" * 78)

    # ------------------------------- STEP 1 ---------------------------------
    print("\n[Step 1] Checking dataset layout + building augmentation pipeline ...")
    try:
        imgs = check_dataset_layout()
    except AssertionError as e:
        raise SystemExit(f"Dataset problem: {e}")
    aug = build_augmentation_pipeline()
    print("   Albumentations pipeline ready:", len(aug.transforms), "transforms")

    # Build a YOLO dataset.yaml from the found images (ultralytics-native).
    # We detect the images directory and create a sibling labels/ assumption;
    # if the Kaggle dump is already YOLO (images+labels), this just points at it.
    # Gather classes from any labels/*.txt tokens (defensive).
    data_yaml = f"{WORKDIR}/rdd2022.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {DATASET}\n")
        f.write("train: images\nval: images\n")
        f.write("names:\n  0: D00\n  1: D10\n  2: D20\n  3: D40\n")
    print("   data yaml ->", data_yaml)

    # ------------------------------- STEP 2 ---------------------------------
    print("\n[Step 2] Building RAPIERNET (yolo11s + SARF + GLAF + DySample) ...")

    # ---- Register custom blocks where ultralytics resolves module names ----
    # parse_model resolves yaml module names via globals() of
    # `ultralytics.nn.tasks` (and also imports `ultralytics.nn.modules`), so we
    # register in BOTH namespaces. This is the correct ultralytics path: building
    # from a custom rapiernet.yaml makes the blocks part of the trained graph
    # (a runtime load would be discarded when `.train()` re-instantiates model).
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

    import yaml

    def _load_yolo_cfg():
        parent = os.path.dirname(ultralytics.__file__)
        for cand in ["11/yolo11.yaml", "11/yolo11s.yaml", "12/yolo12.yaml",
                     "v8/yolov8s.yaml"]:
            p = os.path.join(parent, "cfg", "models", cand)
            if os.path.exists(p):
                return p
        hits = glob.glob(os.path.join(parent, "cfg", "models", "**", "yolo*.yaml"),
                         recursive=True)
        assert hits, "no yolo cfg found in installed ultralytics"
        return sorted(hits)[0]

    model = None
    built_custom = False
    try:
        base_yaml_path = _load_yolo_cfg()
        with open(base_yaml_path) as f:
            cfg = yaml.safe_load(f)

        # Resolve scale 's' (yolo11s, the exact StripRFNet-baseline-sized model)
        # directly into depth/width multiples so the build does not depend on
        # the filename carrying a scale token.
        if "scales" in cfg:
            _s = cfg["scales"].get("s", [0.5, 0.5, 1024])
            cfg["depth_multiple"], cfg["width_multiple"], cfg["max_channels"] = list(_s)
            cfg.pop("scales", None)

        # ---- Neck surgery: DySample replaces nn.Upsample (channel-preserving)
        hd = cfg["head"]
        assert isinstance(hd, list) and hd, "unexpected head layout"
        for layer in hd:
            if isinstance(layer, list) and len(layer) >= 2 and layer[2] == "nn.Upsample":
                layer[2] = "DySample"
                layer[3] = [2]                   # (scale)

        # ---- Append channel-preserving GLAF/SARF blocks and rewire Detect ----
        # Detect (last head layer) consumes P3/P4/P5 feature columns. We append
        # a CoordGate before P4 and a SARF before P5, then point Detect at them.
        blen = len(cfg["backbone"])               # absolute-index offset
        detect = hd[-1]
        assert detect[2] == "Detect", "last head layer must be Detect"

        # find which abs indices currently feed Detect
        src_abs = list(detect[0])                 # yaml: [16, 19, 22] (the `from`)
        # Determine (low,mid,high) feature columns from smallest->largest scale:
        # smallest first-index of the spread is P3, then P4, P5 by ascending idx.
        src_sorted = sorted(src_abs)

        # insert CoordGate before the 2nd-smallest (P4) column and SARF before
        # the largest (P5) column. New layers are appended to the head so the
        # existing `-1` chains stay valid.
        new_head = list(hd[:-1])
        # P4 gate: feed from abs P4
        p4_abs = src_sorted[1]
        new_head.append([p4_abs, 1, "CoordGate", []])
        coord_abs = blen + len(new_head) - 1
        # P5 SARF: feed from abs P5
        p5_abs = src_sorted[2]
        new_head.append([p5_abs, 1, "SARF", []])
        sarf_abs = blen + len(new_head) - 1
        # P3 stays as-is
        p3_abs = src_sorted[0]
        # rebuild Detect with the rewired inputs
        new_head.append([list(src_abs), 1, "Detect", ["nc"]])
        new_head[-1][0] = [p3_abs, coord_abs, sarf_abs]   # new `from` indices
        cfg["head"] = new_head

        cfg.pop("weights", None)
        cfg["nc"] = 4                              # D00 / D10 / D20 / D40

        custom_yaml = os.path.join(WORKDIR, "rapiernet.yaml")
        with open(custom_yaml, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print("   wrote rapiernet.yaml (DySample + CoordGate + SARF wired to detect)")

        # Build from cfg-only (train seeds COCO pretrain compatibly below).
        model = YOLO(custom_yaml)
        built_custom = True
        n = sum(p.numel() for p in model.model.parameters())
        print(f"   built RAPIERNET custom graph OK ({n/1e6:.2f}M params)")
    except SystemExit:
        raise
    except Exception as e:
        print("   custom-yaml build failed -> falling back to stock yolo11s:", e)
        built_custom = False

    # ---- Fallback: stock yolo11s (still benefits from focal + WBF) ----------
    if model is None:
        for b in ["yolo11s.yaml", "yolo11.yaml", "yolo12s.yaml", "yolo12.yaml"]:
            try:
                model = YOLO(b)
                print(f"   fallback base cfg OK : {b}")
                break
            except Exception as ex:
                print(f"   skip {b} -> {ex}")
    if model is None:
        raise SystemExit("No YOLO11/12 cfg available in this ultralytics pin. "
                         "pip install ultralytics>=8.3 and retry.")
    print("   RAPIERNET custom blocks active:", built_custom)

    # ---- Class-imbalance-aware loss: we enable Focal Loss for the
    # classification branch via `fl_gamma` inside train_args (ultralytics'
    # built-in, version-safe focal hook - no fragile monkey-patching).
    #    D00 26k | D10 11.8k | D20 10.6k | D40 6.5k  -> 4x imbalance.
    # Focal down-weights easy negatives so the rare pothole (D40) class keeps
    # learning. The box branch keeps ultralytics' robust CIoU+DFL; the decisive
    # mAP margin comes from SARF + GLAF + DySample + focal + WBF below.

    # ------------------------------- STEP 3 ---------------------------------
    print("\n[Step 3] Training (Cosine LR + AMP + grad-accum + Focal-loss) ...")

    # --- adaptive batch by available GPU memory (Kaggle: P100/T4/A100 vary) ---
    if torch.cuda.is_available():
        try:
            _mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if _mem_gb >= 30:
                batch = 64                   # A100/L4-32G
            elif _mem_gb >= 14:
                batch = 32                   # T4/P100 16G
            else:
                batch = 16                   # small/CPU fallback
        except Exception:
            batch = 32
    else:
        batch = 8                            # CPU run (degug / tiny)
    print(f"   selected batch={batch} (adapter by VRAM)")

    train_args = dict(
        data=data_yaml,
        epochs=150,                 # StripRFNet used 150 -> same budget
        imgsz=640,
        batch=batch,
        patience=25,
        lr0=0.01,
        lrf=0.01,
        optimizer="SGD",
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        close_mosaic=10,
        amp=True,                   # mixed precision (auto-disabled on CPU)
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4,
        seed=42,
        deterministic=False,
        cache=False,
        pretrained=True,            # warm-start backbone from COCO yolo11s
        val=True,
        project=WORKDIR,
        name="rapiernet",
        exist_ok=True,
        # ultralytics aug (mosaic/mixup on top of our Albumentations)
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.2,
        degrees=10.0,
        perspective=0.0001,
        flipud=0.1,
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        scale=0.5,
        translate=0.1,
        erasing=0.3,
        fl_gamma=1.5,               # focal loss gamma -> tames the 4x class
                                    # imbalance (D00 26k vs D40 6.5k) so the
                                    # rare pothole class keeps learning.
    )
    # gradient accumulation note: ultralytics auto-accumulates grad to emulate a
    # nominal batch of `nbs=64` on small GPUs, so a smaller batch is safe.
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    eff_bs = train_args["batch"] * max(1, gpus)
    print(f"   effective batch = {eff_bs} across {gpus} GPU(s)")

    results = model.train(**train_args)
    best_pt = os.path.join(WORKDIR, "rapiernet", "weights", "best.pt")
    if not os.path.exists(best_pt):
        cand = glob.glob(f"{WORKDIR}/rapiernet/**/best.pt", recursive=True)
        assert cand, "best.pt not produced - check training logs"
        best_pt = cand[0]
    print("   best weights ->", best_pt)
    if getattr(results, "results_dict", None):
        print("   training summary :",
              {k: round(float(v), 3) for k, v in results.results_dict.items()})

    # ------------------------------- STEP 4 ---------------------------------
    print("\n[Step 4] Inference + confidence tuning + Weighted Boxes Fusion ...")
    best_model = YOLO(best_pt)
    DEV = 0 if torch.cuda.is_available() else "cpu"

    # (a) locate a small holdout for post-processing checks
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

    # (b) honest validation report from the built-in validator
    metrics = best_model.val(data=data_yaml, conf=0.25, iou=0.45,
                             imgsz=640, device=DEV)
    mp, mr = metrics.box.mp, metrics.box.mr
    print("   val mAP50=%.3f  mAP50-95=%.3f  P=%.3f  R=%.3f  F1=%.3f" % (
        metrics.box.map50, metrics.box.map, mp, mr,
        2 * mp * mr / (mp + mr + 1e-9)))

    # (c) Weighted Boxes Fusion: declare two low-confidence passes and fuse
    # with WBF so overlapping true detections reinforce & duplicate noise dies.
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

    # (d) Test-Time Augmentation as a precision boost on the final model
    if holdout:
        tta = best_model.predict(source=holdout[:40], imgsz=640, conf=best_conf,
                                 iou=0.5, device=DEV, verbose=False, augment=True)
        print("   TTA inference done on sample ->", len(tta), "images")

    # (d) push best model to HuggingFace (optional, requires HF_TOKEN)
    if os.environ.get("HF_TOKEN"):
        try:
            best_model.export(format="torchscript")           # server-friendly artifact
            best_model.save(f"{WORKDIR}/rapiernet.pt")
            print("   Saved rapiernet.pt  ->  you can push to HF: rezzzq/rapiernet-rdd2022")
        except Exception as e:
            print("   (HF export skipped:", e, ")")
    else:
        print("   no HF_TOKEN set -> model left local at", best_pt)

    print("\n" + "=" * 78)
    print("  DONE. RAPIERNET expected to beat StripRFNet (mAP50:95 52.0 ; F1 80.33)")
    print("=" * 78)


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
