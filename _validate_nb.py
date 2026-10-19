import json

nb = json.load(open(r"C:\Users\user\Downloads\final\ablation_china_session.ipynb", encoding="utf-8"))
src = "\n".join("".join(c["source"]) for c in nb["cells"])
print("cells:", len(nb["cells"]))
print("build_variant_2param:", "def build_variant(seed, flags):" in src)
print("model.yaml_swap:", "y.model.yaml = cfg" in src)
print("identity_off:", "layer[2] = 'nn.Identity'" in src)
print("upsample:", "[None, 2, 'nearest']" in src)
print("attr_set:", "newmod.f = getattr(blk, 'f', -1)" in src)
print("comment_marker:", "line-852 rebuild" in src)
print("pre_eval:", "pre-train eval" in src)
assert all(c["cell_type"] in ("markdown", "code") for c in nb["cells"])
print("OK")