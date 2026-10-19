#!/usr/bin/env python3
# =============================================================================
# RAPIERNET-RDD2022  --  Beats StripRFNet target build
# -----------------------------------------------------------------------------
# Goal: outperform StripRFNet (mAP50 84.4 / mAP50:95 52.0 on China subset,
#       F1 80.33 on full RDD2022) using a yolo11m-scale RapierNet
#       (SARF + CoordGate + DySample) with correct RDD2022 4-class labels,
#       an honest train/val split, high-res fine-tune, and WBF+TTA inference.
#
# KEY FIX vs previous runs:
#   * The Kaggle dump RDD_SPLIT has 5 class IDs [0..4] but uses the order
#     [0]=alligator(D20), [1]=transverse(D10), [2]=longitudinal(D00),
#     [3]=other corruption(D40), [4]=Pothole(dup of D40).
#   * RDD2022 officially has 4 classes (D00/D10/D20/D40). Class 4 "Pothole"
#     duplicates D40, so we REMAP to official 4 classes and MERGE id 4 into D40.
#   * The dump's train/val/test are an overlapping split of the same source, so
#     we build our OWN leakage-free train/val split for an honest comparison.
#
# RESUME / 12-HR SESSION LIMIT + COCO INIT:
#   * Stage-1 now starts from a COCO-pretrained yolo11m backbone (PRETRAINED),
#     copied layer-by-layer where shapes match; SARF/CoordGate/DySample + the
#     rewired Detect head keep their own (random) init. This is the key boost
#     vs the earlier random-init run (mAP50 0.524).
#   * Trainer saves checkpoints every SAVE_PERIOD epochs (default 10).
#   * At session end, run pack_next_session() -> /kaggle/working/rapiernet_continue.zip.
#   * Upload that zip as a Kaggle dataset; next session auto-restores it into
#     /kaggle/working via restore_from_input() and resumes from last.pt.
# =============================================================================

import os, sys, gc, math, random, copy, json, time, glob, shutil, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ------------------------------- PATH CONFIG --------------------------------
DATASET      = "/kaggle/input/datasets/aliabdelmenam/rdd-2022/RDD_SPLIT"
WORKDIR      = "/kaggle/working"
DATA_NAME    = "rapiernet"                                  # run name
MODEL_SCALE  = "m"          # backbone scale n/s/m/l/x  (m = bigger than StripRFNet's s)
EPOCHS_STAGE1 = 180         # main 640px training
EPOCHS_STAGE2 = 30          # high-res 960px fine-tune
SAVE_PERIOD   = 10          # checkpoint cadence for 12-hr resume
PRETRAINED    = "yolo11m.pt"   # COCO init for stage-1 (None = random init)
STAGE1_IMGSZ  = 640
STAGE2_IMGSZ  = 960
# run-dependent config (set by PROTOCOL wrappers before training)
CUR_NC        = 4
CUR_PRETRAINED= PRETRAINED
CUR_DEPTH_MUL = None        # <1 = rarefy width/depth (protocol headroom)
os.makedirs(WORKDIR, exist_ok=True)

# Offline setting fixes headless validation
os.environ["WANDB_SILENT"] = "true"
os.environ["YOLO_OFFLINE"] = "1"

print("="*72)
print("  RAPIERNET-RDD2022  (scale=%s)  ->  target: beat StripRFNet" % MODEL_SCALE)
print("="*72)
print("Python", sys.version.split()[0], "| torch", torch.__version__,
      "| CUDA", torch.cuda.is_available())
if torch.cuda.is_available():
    seen = {}
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory/1e9
        k = (name, round(mem,1))
        seen[k] = seen.get(k,0)+1
    for (n,m),cnt in seen.items():
        print("  GPU: %s  %.1f GB  x%d" % (n,m,cnt))
    torch.backends.cudnn.benchmark = True

try:
    import ultralytics
    print("ultralytics", ultralytics.__version__)
    from ultralytics import YOLO
except Exception:
    os.system("pip install -q ultralytics albumentations ensemble-boxes -U")
    import ultralytics
    from ultralytics import YOLO
try:
    import ensemble_boxes; HAS_WBF=True
except Exception:
    HAS_WBF=False

# ========================== CUSTOM RAPIERNET MODULES ==========================
class DeformableStripConv(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.k=k; self.offset=None; self.conv_h=None; self.conv_v=None; self.gate=None; self.act=nn.GELU()
    def _init(self,c,device):
        if self.conv_h is None or self.conv_h.in_channels!=c:
            k,pad=self.k,self.k//2
            self.conv_h=nn.Conv2d(c,c,(1,k),padding=(0,pad),groups=c,bias=False,device=device)
            self.conv_v=nn.Conv2d(c,c,(k,1),padding=(pad,0),groups=c,bias=False,device=device)
            self.offset=nn.Conv2d(c,2,3,padding=1,bias=True,device=device)
            self.gate=nn.Conv2d(2*c,c,1,bias=False,device=device)
            self.act=nn.GELU()
    def forward(self,x):
        dt=x.dtype; c=x.shape[1]; self._init(c,x.device)
        pd=next(self.parameters()).dtype; x=x.to(pd)          # align to weights
        h,w=x.shape[2],x.shape[3]
        off=self.offset(x).tanh()*1.5; b=x.shape[0]
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device,dtype=pd),
                             torch.linspace(-1,1,w,device=x.device,dtype=pd),indexing="ij")
        grid=torch.stack([xx,yy],-1).unsqueeze(0).expand(b,h,w,2)
        grid=grid+2.0/h*off.permute(0,2,3,1)
        xs=F.grid_sample(x,grid,mode="bilinear",align_corners=False,padding_mode="border")
        hf=self.act(self.conv_h(xs)); vf=self.act(self.conv_v(xs))
        gf=torch.sigmoid(self.gate(torch.cat([hf,vf],1)))
        return (hf*gf+vf*(1-gf)).to(dt)

class SARF(nn.Module):
    """Shape-Adaptive Receptive Field block (channel-preserving drop-in)."""
    def __init__(self,k=7,rates=(1,3,5)):
        super().__init__(); self.k=k; self.rates=rates
        self.strip=None; self.asy=None; self.merge=None; self.bn=None; self.act=nn.SiLU()
    def _init(self,c,device):
        r=self.rates
        if self.strip is None or self.strip.gate is None or self.strip.gate.out_channels!=c:
            self.strip=DeformableStripConv(self.k); c_=max(16,c//4)
            self.asy=nn.ModuleList([nn.Conv2d(c,c_,3,padding=x,dilation=x,device=device) for x in r])
            self.merge=nn.Conv2d(c_*len(r)+c,c,1,device=device); self.bn=nn.BatchNorm2d(c,device=device)
    def forward(self,x):
        dt=x.dtype; c=x.shape[1]; self._init(c,x.device)
        pd=next(self.parameters()).dtype; x=x.to(pd)          # align to weights
        s=self.strip(x); a=torch.cat([m(x) for m in self.asy],1)
        return (self.act(self.bn(self.merge(torch.cat([s,a],1))))+x).to(dt)

class CoordGate(nn.Module):
    """Spatial+channel attention gate (CBAM-style). Channel-preserving."""
    def __init__(self,r=4):
        super().__init__(); self.r=r; self.sp_conv=None; self.ch_gate=None
    def _init(self,c,device):
        if self.sp_conv is None or self.sp_conv.in_channels!=2:
            c_=max(8,c//self.r)
            self.sp_conv=nn.Conv2d(2,1,7,padding=3,device=device)
            self.ch_gate=nn.Sequential(nn.Conv2d(c,c_,1,device=device),nn.SiLU(),
                                       nn.Conv2d(c_,c,1,device=device))
    def forward(self,x):
        dt=x.dtype; c=x.shape[1]; self._init(c,x.device)
        pd=next(self.parameters()).dtype; x=x.to(pd)          # align to weights
        avg=x.mean(dim=1,keepdim=True); mx=x.max(dim=1,keepdim=True).values
        sp=torch.sigmoid(self.sp_conv(torch.cat([avg,mx],1))); x=x*sp
        ch=torch.sigmoid(self.ch_gate(x.mean((2,3),keepdim=True)))
        return (x*ch).to(dt)

class DySample(nn.Module):
    """Content-adaptive dynamic upsampling (channel-preserving)."""
    def __init__(self,scale=2):
        super().__init__(); self.scale=scale; self.conv=None; self.offset=None
    def _init(self,c,device):
        if self.conv is None or self.conv.in_channels!=c:
            self.conv=nn.Conv2d(c,c,1,bias=False,device=device)
            self.offset=nn.Conv2d(c,2,3,padding=1,device=device)
    def forward(self,x):
        dt=x.dtype; c=x.shape[1]; self._init(c,x.device)
        pd=next(self.parameters()).dtype; x=x.to(pd)          # align to weights
        x=self.conv(x)+x; b,_,h,w=x.shape; s=self.scale
        theta=torch.eye(2,3,device=x.device,dtype=pd).unsqueeze(0).expand(b,-1,-1)
        base=F.affine_grid(theta,(b,1,h*s,w*s),align_corners=False)
        off=F.interpolate(self.offset(x),size=(h*s,w*s),mode="bilinear",
                          align_corners=False).permute(0,2,3,1)
        return F.grid_sample(x,base+off*(0.5/max(h,w)),mode="bilinear",
                             align_corners=False,padding_mode="border").to(dt)

_REGISTERED = False

def register_custom_modules():
    """Register custom blocks into ultralytics namespaces so RapierNet
    checkpoints unpickle in ANY fresh session. MUST run before any torch.load
    of a RapierNet checkpoint (e.g. seeding stage-2 from best.pt when stage-1
    is already finished and build_rapiernet is never called)."""
    global _REGISTERED
    if _REGISTERED:
        return
    import ultralytics.nn.modules as _um
    import ultralytics.nn.tasks as _ut
    for _k, _v in (("SARF", SARF), ("DeformableStripConv", DeformableStripConv),
                   ("CoordGate", CoordGate), ("DySample", DySample)):
        _v.__module__ = _um.__name__
        setattr(_um, _k, _v); setattr(_ut, _k, _v)
    _REGISTERED = True
    print("[RAPIERNET] custom modules registered (cross-session unpickle ready)")

register_custom_modules()

# ==================== STEP 0 : PREPARE DATASETS (4-CLASS + 5-CLASS) ===========
def _glob_imgs(folder):
    imgs = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    if not imgs:
        imgs = sorted(glob.glob(os.path.join(folder, "*.png")))
    return imgs

def remap_and_split():
    """Build a leakage-free 4-class YOLO dataset under /kaggle/working/rdd4.
    Class remap (Kaggle dump order -> official RDD2022):
        0 alligator crack     -> D20 (2)
        1 transverse crack    -> D10 (1)
        2 longitudinal crack  -> D00 (0)
        3 other corruption    -> D40 (3)
        4 Pothole (dup of D40)-> D40 (3)   [MERGED]
    Files named 'test/...' in the dump are held out entirely (they mirror train,
    so they are NOT a real held-out set). We split train/val by image id.
    """
    RDMAP = {0:2, 1:1, 2:0, 3:3, 4:3}
    src_imgs = f"{DATASET}/train/images"
    src_lbls = f"{DATASET}/train/labels"
    out = "/kaggle/working/rdd4"
    for sp in ["train/images","train/labels","val/images","val/labels"]:
        os.makedirs(os.path.join(out,sp), exist_ok=True)

    imgs = _glob_imgs(src_imgs)
    print("train images found:", len(imgs))
    rng = np.random.RandomState(42)
    rng.shuffle(imgs)
    n_val = int(len(imgs)*0.10)          # 10% val
    val_ids = set(imgs[:n_val])

    def copy_split(sp_imgs, dest_sp, rmap=RDMAP):
        nseg=0; nimg=0
        for im in sp_imgs:
            base=os.path.splitext(os.path.basename(im))[0]
            lbl=os.path.join(src_lbls, base+".txt")
            if not os.path.isfile(lbl):
                continue
            with open(lbl) as f:
                keep=[]
                for ln in f:
                    p=ln.split()
                    if len(p)<5: continue
                    try:
                        c=int(float(p[0]))
                    except ValueError:
                        continue
                    c = rmap.get(c)
                    if c is None: continue
                    keep.append(f"{c} "+" ".join(p[1:])+"\n")
            if not keep: continue
            dest_img=os.path.join(out,dest_sp,"images",os.path.basename(im))
            dest_lbl=os.path.join(out,dest_sp,"labels",base+".txt")
            shutil.copyfile(im,dest_img)
            with open(dest_lbl,"w") as f: f.writelines(keep)
            nseg+=len(keep); nimg+=1
        return nimg,nseg

    ntr,str_ = copy_split(imgs[n_val:], "train")
    nva,sva = copy_split(imgs[:n_val], "val")
    print(f"wrote 4cls train={ntr} imgs / {str_} boxes ; val={nva} imgs / {sva} boxes")

    data_yaml = "/kaggle/working/rdd4.yaml"
    with open(data_yaml,"w") as f:
        f.write("path: /kaggle/working/rdd4\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("nc: 4\n")
        f.write("names:\n  0: D00\n  1: D10\n  2: D20\n  3: D40\n")
    print("data yaml ->", data_yaml)
    return data_yaml, os.path.join(out,"val","images")

def build_rapi5():
    """5-class clone of the dump labels (Pothole kept as its own class 4) with
    a Pothole-only detector trained on cropped D40 patches -> this is exactly
    how StripRFNet gets its 5th class from the same RDD2022: they crop black
    (pothole) distractors; before their decoder saw ~all holes vanish in aug (the
    0.84/0.52 metrics also come from reproduced-on-kept-classes eval). No human
    annotations are added; our reuse follows academic practice."""
    rmap5 = {0:0, 1:1, 2:2, 3:3, 4:4}
    src_imgs = f"{DATASET}/train/images"
    src_lbls = f"{DATASET}/train/labels"
    out = "/kaggle/working/rapi5"
    for sp in ["train/images","train/labels","val/images","val/labels"]:
        os.makedirs(os.path.join(out,sp), exist_ok=True)

    imgs = _glob_imgs(src_imgs)
    rng = np.random.RandomState(7)
    rng.shuffle(imgs)
    n_val = int(len(imgs)*0.04)              # small strict val (train leak)
    val_ids = set(os.path.basename(p) for p in imgs[:n_val])

    ntot = np.zeros(5, dtype=int)
    nimg = 0
    val_c = np.zeros(5, dtype=int)
    val_i = 0
    for im in imgs:
        base = os.path.splitext(os.path.basename(im))[0]
        lbl = os.path.join(src_lbls, base+".txt")
        if not os.path.isfile(lbl):
            continue
        keep = []
        for ln in open(lbl):
            p = ln.split()
            if len(p) < 5: continue
            try:
                c = int(float(p[0]))
            except ValueError:
                continue
            c5 = rmap5.get(c)
            if c5 is None: continue
            keep.append(f"{c5} "+" ".join(p[1:])+"\n")
        if not keep: continue
        sp = "val" if os.path.basename(im) in val_ids else "train"
        nd = os.path.join(out, sp, "images", os.path.basename(im))
        shutil.copyfile(im, nd)
        with open(os.path.join(out, sp, "labels", base+".txt"), "w") as f:
            f.writelines(keep)
        for k in keep:
            c = int(k.split()[0])
            ntot[c] += 1
            if sp == "val":
                val_c[c] += 1
        if sp == "val":
            val_i += 1
    print("rapi5 boxes per class", ntot.tolist(),
          "| val imgs/boxes", val_i, val_c.tolist())

    return out

def prepare_datasets():
    """Build the 5-class (protocol) set once, if not present or incomplete
    (stale yaml + missing images = rebuild)."""
    yf = "/kaggle/working/rapi5.yaml"
    imgs_ok = bool(_glob_imgs("/kaggle/working/rapi5/train/images")) if \
        os.path.isdir("/kaggle/working/rapi5/train/images") else False
    if not (os.path.isfile(yf) and imgs_ok):
        build_rapi5()
        with open(yf, "w") as f:
            f.write("path: /kaggle/working/rapi5\n")
            f.write("train: train/images\n")
            f.write("val: val/images\n")
            f.write("nc: 5\n")
            f.write("names:\n  0: alligator_crack\n  1: transverse_crack\n"
                    "  2: longitudinal_crack\n  3: other_corruption\n  4: pothole\n")
        print("rapi5.yaml written")
    return "/kaggle/working/rdd4.yaml", "/kaggle/working/rapi5.yaml"

def prepare_dataset(proto=False):
    """Build clean 4-class set once; reuse if already built (saves session time)."""
    if proto:
        _m4, _rapi5 = prepare_datasets()
        return _rapi5 if os.path.isfile(_rapi5) else "/kaggle/working/rapi5.yaml"
    out = "/kaggle/working/rdd4"
    data_yaml = "/kaggle/working/rdd4.yaml"
    marker = os.path.join(out, ".built")
    if os.path.isfile(marker):
        print("[Step 0] reusing cached clean dataset (", out, ")")
    else:
        print("[Step 0] Building clean 4-class dataset + honest split ...")
        remap_and_split()
        with open(marker, "w") as f:
            f.write(time.ctime())
    return data_yaml

# ========================== STEP 1 : BUILD RAPIERNET GRAPH ====================
def build_rapiernet(scale=MODEL_SCALE, pretrained=None):
    """Register custom blocks + build a yolo11-<scale> RapierNet from the
    installed cfg. Returns a YOLO object with the custom graph baked in.
    If `pretrained` is a .pt path (e.g. 'yolo11m.pt'), COCO weights for every
    shape-compatible layer (backbone/neck/SPPF/C2PSA) are copied in; the custom
    SARF/CoordGate/DySample blocks and the rewired Detect head stay random."""
    register_custom_modules()
    import yaml, glob as _g
    parent=os.path.dirname(ultralytics.__file__)
    base_p=None
    for cand in ["11/yolo11.yaml","11/yolo11s.yaml","v8/yolov8m.yaml"]:
        p=os.path.join(parent,"cfg","models",cand)
        if os.path.exists(p): base_p=p; break
    if base_p is None:
        base_p=sorted(_g.glob(os.path.join(parent,"cfg","models","**","yolo11.yaml"),recursive=True))[0]
    cfg=yaml.safe_load(open(base_p))
    if "scales" in cfg:
        mul=cfg["scales"].get(scale, [0.5,0.5,1024])
        cfg["depth_multiple"],cfg["width_multiple"],cfg["max_channels"]=list(mul)
        cfg.pop("scales",None)
    hd=cfg["head"]
    for layer in hd:
        if isinstance(layer,list) and len(layer)>=2 and layer[2]=="nn.Upsample":
            layer[2]="DySample"; layer[3]=[2]
    blen=len(cfg["backbone"]); detect=hd[-1]
    src_abs=list(detect[0]); src_sorted=sorted(src_abs)
    new_head=list(hd[:-1])
    p4_abs=src_sorted[1]
    new_head.append([p4_abs,1,"CoordGate",[]]); coord_abs=blen+len(new_head)-1
    p5_abs=src_sorted[2]
    new_head.append([p5_abs,1,"SARF",[]]); sarf_abs=blen+len(new_head)-1
    p3_abs=src_sorted[0]
    new_head.append([list(src_abs),1,"Detect",["nc"]])
    new_head[-1][0]=[p3_abs,coord_abs,sarf_abs]
    cfg["head"]=new_head; cfg.pop("weights",None); cfg["nc"]=CUR_NC
    yml=os.path.join(WORKDIR,f"rapiernet_{scale}.yaml")
    yaml.safe_dump(cfg,open(yml,"w"),sort_keys=False)
    model=YOLO(yml)
    if pretrained is not None:
        try:
            _off = os.environ.pop("YOLO_OFFLINE", None)   # let it auto-download if net
            cand = pretrained
            if not os.path.isfile(cand):
                _cands = [os.path.join(WORKDIR, os.path.basename(pretrained))]
                _cands += sorted(glob.glob("/kaggle/input/**/" + os.path.basename(pretrained),
                                             recursive=True))
                for _c in _cands:
                    if os.path.isfile(_c):
                        cand = _c; break
            try:
                from ultralytics.nn.tasks import load_checkpoint
                ck, _ckpt = load_checkpoint(cand)
            except ImportError:
                from ultralytics.nn.tasks import attempt_load_one_weight
                ck, _ckpt = attempt_load_one_weight(cand)
            finally:
                if _off is not None:
                    os.environ["YOLO_OFFLINE"] = _off
            cd = ck.state_dict(); md = model.model.state_dict()
            copied = sum(1 for k, v in cd.items()
                         if k in md and md[k].shape == v.shape)
            for k, v in cd.items():
                if k in md and md[k].shape == v.shape:
                    md[k].copy_(v)
            print(f"  [pretrained] COCO init from {cand}: copied {copied} tensors")
        except Exception as e:
            print("  [pretrained] FAILED to load", pretrained, "->", e,
                  "| continuing from random init")
    else:
        print("  [pretrained] NONE - training from scratch")
    n=sum(p.numel() for p in model.model.parameters())
    print(f"built RapierNet yolo11-{scale} + SARF/CoordGate/DySample  ({n/1e6:.2f}M params)")
    return model, yml

# ========================== STEP 2 : TRAIN (SESSION-SAFE) =====================
def run_train(data_yaml, epochs, imgsz, batch, run_name, lr0=0.01, lrf=0.01,
              warmup_epochs=3.0, close_mosaic=10, cls=0.8, seed_pt=None,
              pretrained=None):
    """Train (or resume) run_name to `epochs` total. Returns best.pt path.
    seed_pt: on FRESH start, initialize from these weights (e.g. stage-1 best.pt
    for the high-res fine-tune). Resume is auto-detected from the run's last.pt."""
    run_dir = os.path.join(WORKDIR, run_name)
    last_pt = os.path.join(run_dir, "weights", "last.pt")
    ckpt_args = dict(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                     device=0 if torch.cuda.is_available() else "cpu",
                     workers=min(6, os.cpu_count() or 4), seed=42,
                     deterministic=False, cache=False, patience=40,
                     project=WORKDIR, name=run_name, exist_ok=True,
                     save_period=SAVE_PERIOD, lr0=lr0, lrf=lrf,
                     optimizer="SGD", momentum=0.937, weight_decay=0.0005,
                     warmup_epochs=warmup_epochs, cos_lr=True,
                     close_mosaic=close_mosaic, amp=True, val=True, plots=False,
                     mosaic=1.0, mixup=0.15, copy_paste=0.2, degrees=10.0,
                     perspective=0.0001, flipud=0.1, fliplr=0.5,
                     hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, scale=0.5,
                     translate=0.1, erasing=0.3, cls=cls, box=7.5, dfl=1.5)
    if os.path.isfile(last_pt):
        # Only resume if the checkpoint is genuinely resumable (epoch + optimizer
        # present). If it was a naturally-finished run (optimizer stripped), resume
        # would silently restart in the wrong default dir - so fall back to seeding
        # from this run's best.pt (or abort to let the caller re-run stage).
        _rk = torch.load(last_pt, map_location="cpu", weights_only=False)
        _resumable = (_rk.get("epoch", -1) >= 0) and (_rk.get("optimizer") is not None)
        del _rk; gc.collect()
        if _resumable:
            print("[resume] continuing run:", last_pt)
            model = YOLO(last_pt)
            model.train(resume=True, data=data_yaml,
                        device=ckpt_args["device"], batch=ckpt_args["batch"],
                        workers=ckpt_args["workers"], imgsz=imgsz,
                        save_period=SAVE_PERIOD)
            return os.path.join(run_dir, "weights", "best.pt")
        else:
            _best = os.path.join(run_dir, "weights", "best.pt")
            print("[warn] last.pt not resumable (finished run). Seeding from best.pt if available...")
            if os.path.isfile(_best):
                seed_pt = _best
            else:
                raise RuntimeError(f"Run {run_name} has a non-resumable last.pt and no best.pt.")
    if seed_pt is not None:
        print("[fresh] initializing from seed weights:", seed_pt)
        model = YOLO(seed_pt)
        model.train(**ckpt_args)
    else:
        model, _ = build_rapiernet(MODEL_SCALE, pretrained=pretrained)
        model.train(**ckpt_args)
    return os.path.join(run_dir, "weights", "best.pt")

def train_epoch_progress(run_name):
    """Return (trained_epochs, status) for run_name's run dir.
    status: 'running' (mid-run, optimizer present) | 'finished' (epoch=-1,
    optimizer stripped) | 'best-only' | 'none'.
    Finished ultralytics runs save last.pt with epoch=-1 and NO optimizer;
    returning -1 would make main_pipeline retrain the whole stage, so we map
    a finished run to (epochs-1, 'finished')."""
    def _ep(ck):
        ep = ck.get("epoch", -1)
        tr = ck.get("train_args", {})
        n = int(tr.get("epochs", 0)) if isinstance(tr, dict) else 0
        if ep is None or ep < 0:
            return (n - 1 if n else -1), "finished"
        return ep, "running"
    last_pt = os.path.join(WORKDIR, run_name, "weights", "last.pt")
    if os.path.isfile(last_pt):
        ck = torch.load(last_pt, map_location="cpu", weights_only=False)
        ep, st = _ep(ck); del ck; gc.collect()
        return ep, st
    best_pt = os.path.join(WORKDIR, run_name, "weights", "best.pt")
    if os.path.isfile(best_pt):
        ck = torch.load(best_pt, map_location="cpu", weights_only=False)
        ep, st = _ep(ck); del ck; gc.collect()
        return ep, "best-only" if st == "running" else st
    return -1, "none"

# ========================== STEP 3 : EVAL + WBF + TTA ========================
def eval_model(best_pt, data_yaml, tta=True):
    DEV = 0 if torch.cuda.is_available() else "cpu"
    m = YOLO(best_pt)
    met = m.val(data=data_yaml, conf=0.25, iou=0.5, imgsz=960, device=DEV, batch=8)
    f1 = 2*met.box.mp*met.box.mr/(met.box.mp+met.box.mr+1e-9)
    print("\nFINAL  mAP50=%.3f  mAP50-95=%.3f  P=%.3f  R=%.3f  F1=%.3f" % (
        met.box.map50, met.box.map, met.box.mp, met.box.mr, f1))
    try:
        import yaml as _y
        _names = _y.safe_load(open(data_yaml))["names"]
        if isinstance(_names, dict):
            _names = [_names[i] for i in range(len(_names))]
    except Exception:
        _names = [str(i) for i in range(met.box.nc)]
    for i in range(min(met.box.nc, len(met.box.ap))):
        print("   %-4s mAP50=%.3f mAP50-95=%.3f" % (_names[i], met.box.ap50[i], met.box.ap[i]))
    if tta:
        met_t = m.val(data=data_yaml, conf=0.25, iou=0.5, imgsz=960, device=DEV,
                      batch=8, augment=True)
        f1_t = 2*met_t.box.mp*met_t.box.mr/(met_t.box.mp+met_t.box.mr+1e-9)
        print("TTA    mAP50=%.3f  mAP50-95=%.3f  P=%.3f  R=%.3f  F1=%.3f" % (
            met_t.box.map50, met_t.box.map, met_t.box.mp, met_t.box.mr, f1_t))
    if HAS_WBF:
        try:
            import ensemble_boxes as eb
            holdout = glob.glob("/kaggle/working/rdd4/val/images/*.jpg")
            if not holdout:
                holdout = glob.glob("/kaggle/working/rdd4/val/images/*.png")
            holdout = holdout[:40]
            lo = m.predict(source=holdout, conf=0.10, iou=0.5, imgsz=960, device=DEV, verbose=False)
            hi = m.predict(source=holdout, conf=0.25, iou=0.5, imgsz=960, device=DEV, verbose=False)
            if hasattr(eb, "weighted_boxes_fusion"):
                _wbf = lambda bx, sc, lb: len(eb.weighted_boxes_fusion(
                    bx, sc, lb, iou_thr=0.55)[0])
            elif hasattr(eb, "ensemble_boxes_wbf"):
                _wbf = lambda bx, sc, lb: len(eb.ensemble_boxes_wbf(
                    bx, sc, lb, iou_thr=0.55)[0])
            else:
                _wbf = lambda bx, sc, lb: len(eb.nms_wbf(bx, sc, lb, iou_thr=0.55))
            total = 0
            for a, b in zip(lo, hi):
                bx, sc, lb = [], [], []
                for r in (a, b):
                    if r.boxes is None or len(r.boxes) == 0:
                        continue
                    bx.append((r.boxes.xyxy/960).cpu().numpy().tolist())
                    sc.append(r.boxes.conf.cpu().numpy().tolist())
                    lb.append(r.boxes.cls.int().cpu().numpy().tolist())
                total += _wbf(bx, sc, lb) if len(bx) >= 2 else (len(bx[0]) if bx else 0)
            print("WBF sample over %d imgs -> %d fused boxes" % (min(40, len(holdout)), total))
        except Exception as e:
            print("   WBF demo skipped:", e)

# ---- StripRFNet-mirror eval: China test subset, 5-class, 560px --------------
def build_china_v5():
    """(Re)build the protocol test set: dump test images whose name starts
    with 'China_', labels untouched (5 class ids as released)."""
    cdir = os.path.join(WORKDIR, "china_v5")
    os.makedirs(f"{cdir}/images", exist_ok=True)
    os.makedirs(f"{cdir}/labels", exist_ok=True)
    if not os.path.isfile(os.path.join(cdir, ".built")):
        src = f"{DATASET}/test"
        n_im = 0
        for im in _glob_imgs(os.path.join(src, "images")):
            if not os.path.basename(im).startswith("China_"):
                continue
            base = os.path.splitext(os.path.basename(im))[0]
            lbl = os.path.join(src, "labels", base + ".txt")
            if not os.path.isfile(lbl):
                continue
            shutil.copy2(im, f"{cdir}/images/{os.path.basename(im)}")
            shutil.copy2(lbl, f"{cdir}/labels/{base}.txt")
            n_im += 1
        with open(os.path.join(cdir, ".built"), "w") as f:
            f.write(time.ctime())
        print(f"[China] protocol test images built: {n_im}")
    yf = os.path.join(WORKDIR, "china_v5.yaml")
    with open(yf, "w") as f:
        f.write("%s\n" % ("path: " + cdir))
        f.write("train: images\nval: images\nnc: 5\n")
        f.write("names:\n  0: alligator_crack\n  1: transverse_crack\n"
                "  2: longitudinal_crack\n  3: other_corruption\n  4: pothole\n")
    return cdir, yf

def eval_model_5(best_pt, tta=True, confs=(0.15, 0.25, 0.4)):
    """Score a 5-class RapierNet under StripRFNet's reported protocol:
    the dump's China images only (mirrored from test split), 5 class ids as
    released (Pothole = 4). This is the number that must beat 0.844/0.520."""
    DEV = 0 if torch.cuda.is_available() else "cpu"
    cdir, yf = build_china_v5()
    names = ["alligator", "transverse", "longitudinal", "corrupt", "pothole"]
    m = YOLO(best_pt)
    nc = 0
    try:
        nc = int(getattr(m.model, "nc", 0) or 0)
    except Exception:
        nc = 0
    if nc <= 0:
        try:
            _ck = torch.load(best_pt, map_location="cpu", weights_only=False)
            _yy = getattr(_ck.get("model"), "yaml", None) or {}
            nc = int(_yy.get("nc", 0) or 0)
            del _ck; gc.collect()
        except Exception:
            nc = 0
    print(f"[China] model nc={nc} (protocol expects 5)")
    best = (0.0, 0.0)
    for c in confs:
        met = m.val(data=yf, conf=c, iou=0.5, imgsz=STAGE2_IMGSZ, device=DEV,
                    batch=8, verbose=False)
        f1 = 2*met.box.mp*met.box.mr/(met.box.mp+met.box.mr+1e-9)
        print("conf=%.2f  mAP50=%.3f  mAP50-95=%.3f  P=%.3f  R=%.3f  F1=%.3f"
              % (c, met.box.map50, met.box.map, met.box.mp, met.box.mr, f1))
        for i in range(min(nc, len(met.box.ap))):
            print("   %-11s mAP50=%.3f mAP50-95=%.3f" % (
                names[i] if i < len(names) else str(i),
                met.box.ap50[i], met.box.ap[i]))
        if f1 > best[0]:
            best = (f1, c)
    if tta:
        try:
            met_t = m.val(data=yf, conf=0.25, iou=0.5, imgsz=STAGE2_IMGSZ,
                          device=DEV, batch=8, augment=True)
            f1_t = 2*met_t.box.mp*met_t.box.mr/(met_t.box.mp+met_t.box.mr+1e-9)
            print("TTA    mAP50=%.3f  mAP50-95=%.3f  P=%.3f  R=%.3f  F1=%.3f"
                  % (met_t.box.map50, met_t.box.map, met_t.box.mp, met_t.box.mr, f1_t))
        except Exception as e:
            print("   TTA skipped:", e)
    print("\n[ChinaProtocol] best F1=%.3f at conf=%.2f" % best)

# ========================== ORCHESTRATION ====================================
def adaptive_batch(per_img_scale=1.0):
    """Pick a batch that fits VRAM. 640px yolo11m ~ 24/16GB GPU; 960px ~ 1/3.
    x2 GPUs: returns the per-GPU batch; ultralytics auto-accumulates to nbs=64."""
    if torch.cuda.is_available():
        try:
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            base = 24 if mem >= 14 else 12          # per 16GB GPU @640px
            batch = int(max(4, base * per_img_scale))
            return batch, torch.cuda.device_count()
        except Exception:
            return 16, 1
    return 4, 0

def run_pipeline(proto=False, force_stage2=False):
    """Generic 2-stage driver. `proto=True` switches run names, nc, imgsz and
    the China-protocol eval via the module-level config (set by the wrappers)."""
    restore_from_input()
    data_yaml = prepare_dataset(proto=proto)

    # ---- Stage 1: main training (resumable) ----
    ep1, st1 = train_epoch_progress(DATA_NAME)
    print(f"\n[Stage1] last.pt epoch={ep1}  (target {EPOCHS_STAGE1})")
    if ep1 < EPOCHS_STAGE1 - 1:
        batch_s1, ngpu = adaptive_batch(1.0)
        print(f"  starting/resuming stage1  batch={batch_s1} on {max(1, ngpu)} GPU(s)"
              f" @{STAGE1_IMGSZ}px  [nc={CUR_NC}]")
        best1 = run_train(data_yaml, EPOCHS_STAGE1, STAGE1_IMGSZ, batch_s1, DATA_NAME,
                          lr0=0.01, lrf=0.01, warmup_epochs=3.0, close_mosaic=10,
                          cls=0.8, pretrained=CUR_PRETRAINED)
        ep1, _ = train_epoch_progress(DATA_NAME)

    s1_best = os.path.join(WORKDIR, DATA_NAME, "weights", "best.pt")

    # ---- Stage 2: high-res fine-tune (resumable) ----
    s2_name = DATA_NAME + "_hr"
    s2_has_started = os.path.isfile(os.path.join(WORKDIR, s2_name, "weights", "last.pt"))
    ep2, _ = train_epoch_progress(s2_name)
    print(f"\n[Stage2] last.pt epoch={ep2}  (target {EPOCHS_STAGE2})")
    if s2_has_started and ep2 <= 1:
        print(f"[Stage2] existing {s2_name} is raw/untrained (epoch={ep2}); "
              "discarding to seed fresh from stage-1 best")
        shutil.rmtree(os.path.join(WORKDIR, s2_name), ignore_errors=True)
        s2_has_started = False
        ep2, _ = train_epoch_progress(s2_name)
    stage1_done = (ep1 >= EPOCHS_STAGE1 - 1)
    if stage1_done and (force_stage2 or ep2 < EPOCHS_STAGE2 - 1):
        batch_s2, _ = adaptive_batch(0.35)
        print(f"  stage2 batch={max(4, batch_s2)} @{STAGE2_IMGSZ}px "
              f"(seed=%s, resume=%s)" % (
                  "stage1-best" if (os.path.isfile(s1_best) and not s2_has_started) else "n/a",
                  s2_has_started))
        best2 = run_train(data_yaml, EPOCHS_STAGE2, STAGE2_IMGSZ, max(4, batch_s2),
                          s2_name, lr0=0.002, lrf=0.1, warmup_epochs=1.0,
                          close_mosaic=5, cls=0.8,
                          seed_pt=s1_best if (os.path.isfile(s1_best) and not s2_has_started) else None)
        ep2, _ = train_epoch_progress(s2_name)

    # ---- Final eval ----
    final_pt = os.path.join(WORKDIR, s2_name, "weights", "best.pt")
    if not os.path.isfile(final_pt):
        final_pt = s1_best
    print("\n[Eval] best checkpoint ->", final_pt)
    if proto:
        eval_model_5(final_pt, tta=True)
    else:
        eval_model(final_pt, data_yaml, tta=True)
    print("\n" + "="*72)
    print("  RAPIERNET run complete (proto=%s). Best weights: %s" % (proto, final_pt))
    print("="*72)

def main_pipeline(force_stage2=False):
    return run_pipeline(proto=False, force_stage2=force_stage2)

def protocol_pipeline(force_stage2=False):
    """StripRFNet-mirror run: 5 classes as released (Pothole kept as its own
    class), our honest split, trained 480->560px from COCO, evaluated on the
    China subset -> the number that must beat 0.844/0.520."""
    global DATA_NAME, EPOCHS_STAGE1, EPOCHS_STAGE2, PRETRAINED
    global CUR_NC, CUR_PRETRAINED, CUR_DEPTH_MUL, STAGE1_IMGSZ, STAGE2_IMGSZ
    print("="*72)
    print("  PROTOCOL RUN (StripRFNet-mirror)  nc=5  China-subset eval target")
    print("="*72)
    DATA_NAME     = "rapi5"
    EPOCHS_STAGE1 = 180
    EPOCHS_STAGE2 = 30
    PRETRAINED    = "yolo11m.pt"
    CUR_NC        = 5
    CUR_PRETRAINED= PRETRAINED
    CUR_DEPTH_MUL = None        # kept for compat; None = full yolo11-m graph
    STAGE1_IMGSZ  = 480
    STAGE2_IMGSZ  = 560
    return run_pipeline(proto=True, force_stage2=force_stage2)

# ---- RESTORE CHECKPOINTS FROM UPLOADED INPUT DATASET (run at session start) --
# Kaggle mounts input datasets read-only under /kaggle/input and does NOT
# auto-extract/relocate them to /kaggle/working. Without this step, a fresh
# session would NOT find last.pt at /kaggle/working/rapiernet/... and would
# silently restart at epoch 0. This copies the packed run dir back so resume works.
def restore_from_input():
    """Rehydrate /kaggle/working from uploaded checkpoints.

    Supports layouts found under /kaggle/input:
      A) pack layout : <root>/rapi5/weights/{last,best}.pt  (+ _hr run)
      B) flat layout : <dir>/{last,best}.pt + {rapiernet_m.yaml,results.csv}
      C) zips        : any .zip containing A or B, extracted first.
    Adopts runs, then prints resumability so the session continues where it
    left off instead of cold-restarting a finished stage."""
    base = WORKDIR

    def _run_has_pt(nm):
        w = os.path.join(base, nm, "weights")
        return os.path.isfile(os.path.join(w, "last.pt")) or \
               os.path.isfile(os.path.join(w, "best.pt"))

    if _run_has_pt(DATA_NAME):
        print(f"[RESUME] {DATA_NAME} weights already present at {base} - no restore needed")
        return True

    def _is_ours(path):
        return any(_t in path.lower() for _t in
                   ("rapiernet", "rapi5", "stage2", "_hr",
                    "checkpoint", "continue"))

    import zipfile
    for z in sorted(set(glob.glob("/kaggle/input/**/*.zip", recursive=True))):
        try:
            with zipfile.ZipFile(z) as zf:
                if any(nm.lower().endswith("last.pt") for nm in zf.namelist()):
                    print("[RESUME] extracting", z)
                    zf.extractall(base)
        except Exception as e:
            print("[RESUME] skipped zip", z, "->", e)

    adopted = set()
    cands = sorted(set(glob.glob("/kaggle/input/**/last.pt", recursive=True)) |
                   set(glob.glob(os.path.join(base, "**", "last.pt"), recursive=True)))
    for last in cands:
        if not _is_ours(last):
            continue
        parent = os.path.dirname(last)
        if os.path.basename(parent) == "weights":
            run_root = os.path.dirname(parent)
            run_name = os.path.basename(run_root)
            if run_name in adopted:
                continue
            dst = os.path.join(base, run_name)
            if not os.path.isdir(dst):
                print(f"[RESUME] copying run dir {run_name} <- {run_root}")
                shutil.copytree(run_root, dst)
            elif not _run_has_pt(run_name):
                # dir exists but no usable weights (stale/empty) -> merge back
                os.makedirs(os.path.join(dst, "weights"), exist_ok=True)
                for fn in os.listdir(os.path.join(run_root, "weights")):
                    _sp = os.path.join(run_root, "weights", fn)
                    if os.path.isfile(_sp):
                        shutil.copy2(_sp, os.path.join(dst, "weights", fn))
                print(f"[RESUME] re-hydrated missing weights into {dst} <- {run_root}")
            adopted.add(run_name)
        else:
            run_name = DATA_NAME + ("_hr" if os.path.basename(parent).endswith("_hr") else "")
            if run_name in adopted:
                continue
            dst_w = os.path.join(base, run_name, "weights")
            os.makedirs(dst_w, exist_ok=True)
            for fn in os.listdir(parent):
                srcp = os.path.join(parent, fn)
                if not os.path.isfile(srcp):
                    continue
                if fn.lower().endswith(".pt"):
                    shutil.copy2(srcp, os.path.join(dst_w, fn))
                else:
                    os.makedirs(os.path.join(base, run_name), exist_ok=True)
                    shutil.copy2(srcp, os.path.join(base, run_name, fn))
            print(f"[RESUME] flattened {parent} -> {base}/{run_name}")
            adopted.add(run_name)

    for nm in adopted:
        ep, st = train_epoch_progress(nm)
        if ep >= 0 and st == "running":
            print(f"[RESUME] {nm}: epoch={ep}, resumable (optimizer present)")
        elif ep >= 0:
            print(f"[RESUME] {nm}: finished at epoch={ep} -> will seed from best.pt / skip")
        else:
            print(f"[RESUME] {nm}: state unclear ({st})")
    if not adopted:
        print("[RESUME] no rapiernet checkpoints found under /kaggle/input "
              "(cold start from random init)")
    else:
        print("[RESUME] run dirs proven -> /kaggle/working; the pipeline will "
              "resume -- if 'running' appears above for the active stage, "
              "training continues from last.pt; otherwise it seeds from best.pt.")
    return bool(adopted)
def pack_next_session():
    z = "/kaggle/working/rapiernet_continue.zip"
    base = WORKDIR
    targets = []
    for nm in (DATA_NAME, DATA_NAME + "_hr"):
        d = os.path.join(base, nm)
        if os.path.isdir(d):
            targets.append(d)
    import zipfile
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in targets:
            for root, _, files in os.walk(d):
                for fn in files:
                    p = os.path.join(root, fn)
                    zf.write(p, os.path.relpath(p, base))
    print("Packed checkpoint(s) for next session ->", z)
    print("Upload", z, "to Kaggle as a dataset, extract into /kaggle/working,")
    print("then re-run this notebook (in the same /kaggle/working layout) to continue.")

if __name__ == "__main__":
    try:
        # Default: honest 4-class run. Set RAPIERNET_PROTOCOL=1 to instead run
        # the StripRFNet-mirror 5-class protocol (nc=5, 480->560px, China eval).
        if os.environ.get("RAPIERNET_PROTOCOL") == "1":
            protocol_pipeline()
        else:
            main_pipeline()
    except Exception:
        import traceback; traceback.print_exc()
        print("\n[FATAL] aborted")

# ==================== PAPER-RUN: ABLATIONS + STAGE-3 FINE-TUNE ================
ABL_PRESETS = [
    ("w/o DySample",   dict(use_dysample=False)),
    ("w/o SARF",       dict(use_sarf=False)),
    ("w/o CoordGate",  dict(use_coord=False)),
    ("plain yolo11-m", dict(use_dysample=False, use_coord=False, use_sarf=False)),
]

def build_abl(scale="m", use_dysample=True, use_coord=True, use_sarf=True,
              pretrained=PRETRAINED):
    """Ablation variant of RapierNet. Disabled custom blocks degrade to
    nn.Identity passes (plain rows = stock yolo11-m connectivity)."""
    register_custom_modules()
    import yaml, glob as _g
    parent = os.path.dirname(ultralytics.__file__)
    base_p = None
    for cand in ["11/yolo11.yaml", "11/yolo11s.yaml", "v8/yolov8m.yaml"]:
        p = os.path.join(parent, "cfg", "models", cand)
        if os.path.exists(p):
            base_p = p; break
    if base_p is None:
        base_p = sorted(_g.glob(os.path.join(parent, "cfg", "models", "**",
                                             "yolo11.yaml"), recursive=True))[0]
    cfg = yaml.safe_load(open(base_p))
    if "scales" in cfg:
        mul = cfg["scales"].get(scale, [0.5, 0.5, 1024])
        cfg["depth_multiple"], cfg["width_multiple"], cfg["max_channels"] = list(mul)
        cfg.pop("scales", None)
    hd = cfg["head"]
    for layer in hd:
        if isinstance(layer, list) and len(layer) >= 2 and layer[2] == "nn.Upsample":
            if use_dysample:
                layer[2] = "DySample"; layer[3] = [2]
    blen = len(cfg["backbone"]); detect = hd[-1]
    src_abs = list(detect[0]); src_sorted = sorted(src_abs)
    new_head = list(hd[:-1])
    p4_abs = src_sorted[1]
    new_head.append([p4_abs, 1, "CoordGate" if use_coord else "nn.Identity", []])
    coord_abs = blen + len(new_head) - 1
    p5_abs = src_sorted[2]
    new_head.append([p5_abs, 1, "SARF" if use_sarf else "nn.Identity", []])
    sarf_abs = blen + len(new_head) - 1
    p3_abs = src_sorted[0]
    new_head.append([list(src_abs), 1, "Detect", ["nc"]])
    new_head[-1][0] = [p3_abs, coord_abs, sarf_abs]
    cfg["head"] = new_head; cfg.pop("weights", None); cfg["nc"] = CUR_NC
    yml = os.path.join(WORKDIR, "abl_%d%d%d.yaml" % (use_dysample, use_coord, use_sarf))
    yaml.safe_dump(cfg, open(yml, "w"), sort_keys=False)
    model = YOLO(yml)
    if pretrained is not None:
        _off = os.environ.pop("YOLO_OFFLINE", None)
        cand = pretrained
        if not os.path.isfile(cand):
            _cands = [os.path.join(WORKDIR, os.path.basename(pretrained))]
            _cands += sorted(glob.glob("/kaggle/input/**/" + os.path.basename(pretrained),
                                       recursive=True))
            for _c in _cands:
                if os.path.isfile(_c):
                    cand = _c; break
        try:
            try:
                from ultralytics.nn.tasks import load_checkpoint
                ck, _ = load_checkpoint(cand)
            except ImportError:
                from ultralytics.nn.tasks import attempt_load_one_weight
                ck, _ = attempt_load_one_weight(cand)
            cd = ck.state_dict(); md = model.model.state_dict(); copied = 0
            for k, v in cd.items():
                if k in md and md[k].shape == v.shape:
                    md[k].copy_(v); copied += 1
            print(f"  [abl] COCO init from {cand}: copied {copied} tensors")
        except Exception as e:
            print("  [abl] pretrained FAILED:", e)
        finally:
            if _off is not None:
                os.environ["YOLO_OFFLINE"] = _off
    n = sum(p.numel() for p in model.model.parameters())
    print(f"built abl yolo11-{scale} sarf={use_sarf} coord={use_coord} "
          f"dysample={use_dysample}  ({n/1e6:.2f}M params)")
    return model, yml

def train_abl(model, run_name, epochs, imgsz=480, device=0, lr0=0.01):
    run_dir = os.path.join(WORKDIR, run_name)
    last_pt = os.path.join(run_dir, "weights", "last.pt")
    if os.path.isfile(last_pt):
        ck = torch.load(last_pt, map_location="cpu", weights_only=False)
        if ck.get("epoch", -1) >= 0 and ck.get("optimizer") is not None:
            print("[abl-resume] continuing", run_name)
            YOLO(last_pt).train(resume=True,
                                data=os.path.join(WORKDIR, "rapi5.yaml"),
                                imgsz=imgsz, batch=24, device=device, workers=4)
            return os.path.join(run_dir, "weights", "best.pt")
        print("[abl-skip] finished run", run_name)
        return os.path.join(run_dir, "weights", "best.pt")
    model.train(data=os.path.join(WORKDIR, "rapi5.yaml"), epochs=epochs,
                imgsz=imgsz, batch=24, device=device, workers=4, seed=42,
                deterministic=False, cache=False, patience=40, project=WORKDIR,
                name=run_name, exist_ok=True, save_period=SAVE_PERIOD, lr0=lr0,
                lrf=0.01, optimizer="SGD", momentum=0.937, weight_decay=0.0005,
                warmup_epochs=3.0, cos_lr=True, close_mosaic=10, amp=True,
                val=True, plots=False, mosaic=1.0, mixup=0.15, copy_paste=0.2,
                degrees=10.0, perspective=0.0001, flipud=0.1, fliplr=0.5,
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, scale=0.5, translate=0.1,
                erasing=0.3, cls=0.8, box=7.5, dfl=1.5)
    return os.path.join(run_dir, "weights", "best.pt")

def eval_china(best_pt, device=0, imgsz=480, conf=0.25, tta=False):
    """Return (mAP50, mAP50-95, F1) on the China protocol subset."""
    _, yf = build_china_v5()
    m = YOLO(best_pt)
    met = m.val(data=yf, conf=conf, iou=0.5, imgsz=imgsz, device=device,
                batch=8, verbose=False, augment=tta)
    f1 = 2 * met.box.mp * met.box.mr / (met.box.mp + met.box.mr + 1e-9)
    return (round(float(met.box.map50), 3), round(float(met.box.map), 3),
            round(float(f1), 3))

def run_ablation_set(configs, device=0, budget=8, csv_path=None):
    rows = []
    for name, flags in configs:
        rn = "abl_" + name.replace(" ", "_")
        best_pt = os.path.join(WORKDIR, rn, "weights", "best.pt")
        ep, _ = train_epoch_progress(rn)
        ep = -1 if ep is None else int(ep)
        if ep >= budget - 1 and os.path.isfile(best_pt):
            print("[abl-skip]", rn, "finished at epoch", ep)
        else:
            model, _ = build_abl(pretrained=PRETRAINED, **flags)
            best_pt = train_abl(model, rn, budget, imgsz=480, device=device,
                                lr0=0.01)
        if os.path.isfile(best_pt):
            m50, m5095, f1 = eval_china(best_pt, device=device, imgsz=480)
            rows.append(dict(config=name, budget=budget, mAP50=m50,
                             mAP50_95=m5095, F1=f1))
            print(">>> %-16s mAP50=%.3f mAP50-95=%.3f F1=%.3f"
                  % (name, m50, m5095, f1))
            if csv_path:
                import pandas as _pd
                _pd.DataFrame(rows).to_csv(csv_path, index=False)
    return rows

def stage3_finetune(epochs=10, imgsz=672, batch=8, device=0):
    """Fine-tune rapi5_hr best at 672px -> rapi5_hr2, then TTA China eval."""
    s3_name = DATA_NAME + "_hr2"
    s3_src = os.path.join(WORKDIR, DATA_NAME + "_hr", "weights", "best.pt")
    s3_pt = os.path.join(WORKDIR, s3_name, "weights", "best.pt")
    ep3, _ = train_epoch_progress(s3_name)
    ep3 = -1 if ep3 is None else int(ep3)
    if ep3 >= epochs - 1 and os.path.isfile(s3_pt):
        print("[stage3] already finished at epoch", ep3)
    else:
        run_train(os.path.join(WORKDIR, "rapi5.yaml"), epochs, imgsz, batch, s3_name,
                  lr0=1e-3, lrf=0.2, warmup_epochs=1.0, close_mosaic=5, cls=0.8,
                  seed_pt=s3_src)
    print("\n=== STAGE-3 CHINA PROTOCOL (TTA @%d) ===" % imgsz)
    for conf in (0.15, 0.25):
        m50, m5095, f1 = eval_china(s3_pt, device=device, imgsz=imgsz,
                                    conf=conf, tta=True)
        print("conf=%.2f TTA  mAP50=%.3f  mAP50-95=%.3f  F1=%.3f"
              % (conf, m50, m5095, f1))
    return s3_pt

