#!/usr/bin/env python3
# INSPECT_RDD_SPLIT.py -- run this cell on Kaggle to learn the REAL class layout
# of your /kaggle/input/datasets/aliabdelmenam/rdd-2022/RDD_SPLIT labels.
import os, glob
from collections import Counter

DATASET = "/kaggle/input/datasets/aliabdelmenam/rdd-2022/RDD_SPLIT"

# 1) Find label files (.txt) in the dump. They live next to images/ as labels/.
txts = glob.glob(f"{DATASET}/**/labels/**/*.txt", recursive=True)
if not txts:
    # Some dumps name it 'annotations' or put txt in same tree
    txts = glob.glob(f"{DATASET}/**/*.txt", recursive=True)
print("label files found:", len(txts))
for t in txts[:5]:
    print("  sample:", t)

# 2) Count distinct class IDs actually present and per-class instance counts
cls_counter = Counter()
bad_lines = 0
nonalpha = 0
for t in txts[:4000]:            # sample first 4000 files (fast)
    try:
        with open(t) as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 5:
                    bad_lines += 1
                    continue
                try:
                    c = int(float(parts[0]))
                except ValueError:
                    nonalpha += 1
                    continue
                cls_counter[c] += 1
    except Exception:
        bad_lines += 1

print("\ndistinct class IDs found in labels:", sorted(cls_counter))
print("per-class instance counts (sampled):")
for c, n in sorted(cls_counter.items()):
    print(f"   class {c}: {n}")

# 3) Show a couple raw label lines to eyeball the class numbering convention
print("\nraw label snippets:")
shown = 0
for t in txts[:200]:
    with open(t) as f:
        line = f.readline().strip()
    if line:
        print("  ", os.path.basename(t), "->", line)
        shown += 1
        if shown >= 5:
            break

# 4) Split layout
for part in ["train", "val", "test", "images"]:
    imgs = glob.glob(f"{DATASET}/{part}/images/**/*.jpg", recursive=True)
    print(f"\n{part}/images jpg count:", len(imgs))

print("\nDone. Report these numbers back.")
