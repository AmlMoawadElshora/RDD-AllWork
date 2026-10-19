import json
import os

OUT = r"C:\Users\user\Downloads\final\resume_stage3_figs.ipynb"

W = "/kaggle/working"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}


cells = []

cells.append(md([
    "# Resume `rapi5_hr2` fine-tune + regenerate paper figures\n\n",
    "Use this notebook when a run was **stopped mid-training** (or killed by session limits).\n\n",
    "It restores the checkpoints/configs you uploaded into `/kaggle/working`,"
    " **resumes the 10-epoch stage-3 fine-tune from its `last.pt`**"
    " (no epoch is lost), then regenerates every paper figure"
    " (`fig_pr_curve`, `fig_f1_curve`, `fig_confusion_matrix`, `fig5_attention_comparison`,"
    " `fig_qualitative`, ablation bars, architecture schematic).\n\n",
    "Run cells **top to bottom in one session** (silver T4 / P100, ~75 min total: 9 epochs resume + figures)."
]))

cells.append(md([
    "## 0) Upload requirements\n\n",
    "As a **Kaggle Dataset** (e.g. `new-dataset` → upload a single zip named `rapiernet_resume.zip`), `+ Add input` below, then run everything else. The zip must contain:\n\n",
    "```\nrapi5_hr2/                      # the interrupted run dir you downloaded\n"
    "  args.yaml   results.csv\n"
    "  weights/last.pt   weights/best.pt\n"
    "rapi5_hr/                       # stage-2 source (from rapi5_hr downloads)\n"
    "  args.yaml\n"
    "  weights/last.pt  weights/best.pt     # rename 'best (4).pt'-style files to best.pt\n"
    "ablation_summary.csv\n"
    "rapiernet_rdd2022_final.py\n"
    "make_paper_figures.py\n"
    "```\n\n",
    "Also `+ Add input` the **RDD-2022 dataset** (the same one used to build the sub-splits).\n\n",
    "Notably *nothing* re-trains from zero — if `last.pt` is missing the resume cell aborts with a clear message instead of silently restarting."
]))

cells.append(code([
    "# 1) dependencies -----------------------------------------------\n",
    "import subprocess, sys\n",
    "def _have(pkg):\n",
    "    return subprocess.run([sys.executable, '-c', f'import {pkg}'],\n",
    "                          capture_output=True).returncode == 0\n",
    "if not _have('ultralytics') or not _have('cv2'):\n",
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n",
    "                           '--no-cache-dir', 'ultralytics==8.3.283'])\n",
    "if not _have('yaml'):\n",
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pyyaml'])\n",
    "print('deps ok')\n"
]))

cells.append(code([
    "# 2) restore checkpoints / configs into /kaggle/working ----------\n",
    "import glob, os, shutil, zipfile\n",
    "from pathlib import Path\n",
    "W = Path('/kaggle/working'); W.mkdir(exist_ok=True)\n",
    "\n",
    "# unpack anything uploaded as a dataset (zip layout above)\n",
    "for z in sorted(glob.glob('/kaggle/input/**/*.zip', recursive=True)):\n",
    "    print('unpack', os.path.basename(z))\n",
    "    with zipfile.ZipFile(z) as zf:\n",
    "        zf.extractall(str(W))\n",
    "\n",
    "# adopt any run dir (named *rapi5* / *_hr* / *stage*) that carries weights/*.pt\n",
    "runs = set()\n",
    "for wd in glob.glob(str(W / '**' / 'weights'), recursive=True):\n",
    "    run = os.path.basename(os.path.dirname(wd))\n",
    "    if not any(t in run for t in ('rapi5', '_hr', 'stage', 'abl', 'Rapiernet')):\n",
    "        continue\n",
    "    pick = {}\n",
    "    for p in sorted(os.listdir(wd)):\n",
    "        low = p.lower()\n",
    "        if not low.endswith('.pt'):\n",
    "            continue\n",
    "        if 'amp' in low or low.count('.pt') > 1:\n",
    "            continue\n",
    "        if low.startswith('last'):\n",
    "            pick['last.pt'] = os.path.join(wd, p)\n",
    "        elif low.startswith('best'):\n",
    "            pick['best.pt'] = os.path.join(wd, p)\n",
    "    for dst, src in pick.items():\n",
    "        if src and os.path.basename(src) != dst:\n",
    "            print(f'  {run}: {os.path.basename(src)} -> {dst}')\n",
    "            shutil.copy2(src, os.path.join(wd, dst))\n",
    "    if pick:\n",
    "        runs.add(run)\n",
    "    print('  run dir ok:', run, sorted(os.path.basename(p) for p in pick))\n",
    "\n",
    "# restore args.yaml next to each adopted run if missing\n",
    "for a in glob.glob(str(W / '**' / 'args*.yaml'), recursive=True):\n",
    "    run = os.path.basename(os.path.dirname(a))\n",
    "    target = os.path.join(str(W), run, 'args.yaml')\n",
    "    if run in runs and not os.path.exists(target) and os.path.abspath(a) != os.path.abspath(target):\n",
    "        shutil.copy2(a, target)\n",
    "        print('  args:', os.path.basename(a), '->', target)\n",
    "\n",
    "# pull scripts / csv to working root (dedupe)\n",
    "for pat in ('rapiernet_rdd2022_final.py', 'make_paper_figures.py', 'ablation_summary.csv'):\n",
    "    for f in glob.glob(str(W / '**' / pat), recursive=True):\n",
    "        dst = str(W / os.path.basename(f))\n",
    "        if os.path.abspath(f) != os.path.abspath(dst) and not os.path.exists(dst):\n",
    "            shutil.copy2(f, dst)\n",
    "            print('  adopted', os.path.basename(f))\n",
    "\n",
    "print('\\nrequired artifacts:')\n",
    "for f in [W/'rapiernet_rdd2022_final.py', W/'make_paper_figures.py',\n",
    "          W/'ablation_summary.csv', W/'rapi5_hr2'/'weights'/'last.pt',\n",
    "          W/'rapi5_hr2'/'args.yaml']:\n",
    "    print(('  OK  ' if f.exists() else '  MISSING '), f)\n",
    "assert (W/'rapiernet_rdd2022_final.py').exists(), 'protocol script missing'\n",
    "assert (W/'rapi5_hr2'/'weights'/'last.pt').exists(), 'rapi5_hr2/last.pt missing'\n"
]))

cells.append(code([
    "# 3) environment ------------------------------------------------\n",
    "import torch, ultralytics\n",
    "print('cuda available:', torch.cuda.is_available())\n",
    "if torch.cuda.is_available():\n",
    "    print('gpu:', torch.cuda.get_device_name(0))\n",
    "print('torch', torch.__version__, '| ultralytics', ultralytics.__version__)\n"
]))

cells.append(code([
    "# 4) load protocol (registers custom modules + dataset builders) --\n",
    "import runpy, os\n",
    "from pathlib import Path\n",
    "W = Path('/kaggle/working')\n",
    "ns = runpy.run_path(str(W / 'rapiernet_rdd2022_final.py'), run_name='not_main')\n",
    "# keep experiment constants the same as the original run\n",
    "ns['DATA_NAME'] = 'rapi5'; ns['CUR_NC'] = 5\n",
    "ns['STAGE1_IMGSZ'] = 480; ns['STAGE2_IMGSZ'] = 560\n",
    "print('protocol constants:', ns.get('DATA_NAME'), ns.get('CUR_NC'),\n",
    "      '| imgsz1/2:', ns.get('STAGE1_IMGSZ'), ns.get('STAGE2_IMGSZ'))\n"
]))

cells.append(code([
    "# 5) build RAPI-5 + China v5 datasets (idempotent) -----------------\n",
    "import glob, os\n",
    "W = Path('/kaggle/working')\n",
    "\n",
    "cand = next(iter(glob.glob('/kaggle/input/**/RDD_SPLIT', recursive=True)[:1] or []), None)\n",
    "ds = os.path.abspath(ns['DATASET'])\n",
    "if not os.path.isdir(ds) and cand:\n",
    "    ns['DATASET'] = cand\n",
    "    print('dataset root ->', cand)\n",
    "assert os.path.isdir(ns['DATASET']) or os.path.isdir(ds), \\\n",
    "    'RDD-2022 split not found; add the dataset input'\n",
    "\n",
    "_m4, _rapi5 = ns['prepare_datasets']()\n",
    "print('prepared:', _m4, '|', _rapi5)\n",
    "print('china yaml:', (W / 'china_v5.yaml').resolve() if (W / 'china_v5.yaml').exists() else 'MISSING')\n",
    "assert (W / 'rapi5.yaml').exists() and (W / 'china_v5.yaml').exists()\n"
]))

cells.append(code([
    "# 6) RESUME stage-3 fine-tune (10 epochs, 672 px) -------------------\n",
    "# Uses run_train -> auto-resume: it verifies epoch>=0 + optimizer present in\n",
    "# rapi5_hr2/weights/last.pt, then continues exactly where the last session stopped.\n",
    "import time, os, glob\n",
    "from pathlib import Path\n",
    "W = Path('/kaggle/working')\n",
    "\n",
    "# sanity: the resume inputs must be present BEFORE the run starts\n",
    "need = [W/'rapi5_hr2'/'weights'/'last.pt', W/'rapi5_hr2'/'weights'/'best.pt',\n",
    "        W/'rapi5_hr2'/'args.yaml', W/'rapi5_hr2'/'results.csv']\n",
    "for f in need:\n",
    "    print('found ' if f.exists() else 'MISSING', f)\n",
    "assert all(f.exists() for f in need)\n",
    "\n",
    "t0 = time.time()\n",
    "best3 = ns['stage3_finetune'](epochs=10, imgsz=672, batch=8, device=0)\n",
    "print('\\nstage-3 resumed + finished in %.1f min' % ((time.time() - t0) / 60))\n",
    "print('best checkpoint:', best3)\n"
]))

cells.append(code([
    "# 7) regenerate all paper figures ------------------------------------\n",
    "import os, glob\n",
    "from pathlib import Path\n",
    "W = Path('/kaggle/working')\n",
    "os.chdir(str(W))\n",
    "import matplotlib\n",
    "matplotlib.use('Agg')\n",
    "import make_paper_figures as mpf\n",
    "\n",
    "OUT = str(W / 'paper_figs'); os.makedirs(OUT, exist_ok=True)\n",
    "CHINA = str(W / 'china_v5.yaml')\n",
    "BEST = str(W / 'rapi5_hr2' / 'weights' / 'best.pt')\n",
    "assert os.path.isfile(BEST), 'rapi5_hr2/weights/best.pt missing after resume'\n",
    "assert (W / 'ablation_summary.csv').exists(), 'ablation_summary.csv missing'\n",
    "print('using', BEST)\n",
    "\n",
    "# one China-subset val -> reused by PR / F1 / confusion-matrix figures\n",
    "m, met = mpf._val_metrics(BEST, CHINA, device=0)   # ~2-3 min\n",
    "mpf.fig_pr_curve(met, OUT)\n",
    "mpf.fig_f1_curve(met, OUT)\n",
    "mpf.fig_confusion(met, OUT)\n",
    "\n",
    "# GT vs prediction + attention comparison (crack/pothole-priority sampling)\n",
    "mpf.fig_samples(BEST, CHINA, '', OUT, split='test')\n",
    "mpf.fig_attention(BEST, CHINA, '', OUT, base_pt=None, split='test')\n",
    "\n",
    "# ablation bars + architecture schematic\n",
    "mpf.fig_ablation(str(W / 'ablation_summary.csv'), OUT)\n",
    "mpf.fig_architecture(OUT)\n",
    "\n",
    "print('\\ndone. figures in', OUT)\n",
    "for f in sorted(glob.glob(OUT + '/*.*')):\n",
    "    print('  ', os.path.basename(f))\n"
]))

cells.append(md([
    "## 8) Collecting the output\n\n",
    "Everything you need is under `/kaggle/working`:\n\n",
    "- `rapi5_hr2/` — the finished 10-epoch run (`best.pt`, `last.pt`, `args.yaml`, `results.csv`)\n",
    "- `paper_figs/` — all regenerated figures (PDF + PNG)\n\n",
    "Optionally pack them into one file before downloading:\n",
    "```python\n",
    "import zipfile, os, glob\n",
    "src = '/kaggle/working'\n",
    "have = [p for p in ('rapi5_hr2', 'paper_figs', 'ablation_summary.csv') if os.path.exists(os.path.join(src, p))]\n",
    "zf = zipfile.ZipFile('/kaggle/working/output_resume.zip', 'w', zipfile.ZIP_DEFLATED)\n",
    "for name in have:\n",
    "    for root, _, files in os.walk(os.path.join(src, name)):\n",
    "        for f in files:\n",
    "            fp = os.path.join(root, f)\n",
    "            zf.write(fp, os.path.relpath(fp, src))\n",
    "zf.close()\n",
    "print('zipped:', have)\n",
    "```\n"
]))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", OUT, os.path.getsize(OUT), "bytes")