import os, sys, numpy as np, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath("33_k_dyn.py")))
import importlib.util
s = importlib.util.spec_from_file_location("kd", "33_k_dyn.py"); kd = importlib.util.module_from_spec(s); s.loader.exec_module(kd)
OUT = kd.OUT; mdl = joblib.load(os.path.join(OUT, "k_dyn.joblib"))["model"]
def f1(G, c, h, m):
    tp, fp, fn = kd.at_c(G, c)[h]; tp, fp, fn = tp[m].sum(), fp[m].sum(), fn[m].sum(); return 2*tp/max(2*tp+fp+fn, 1)
res = {}
for part in ("val", "test"):
    z = np.load(os.path.join(OUT, f"k_dyn_cache_{part}.npz")); G = (z["A"], z["TP"], z["ACT"]); X = z["X"]
    grow = G[2][:, 1] >= 4; raw = np.exp(mdl.predict(X)); res[part] = (G, raw, grow)
Gv, rv, gv = res["val"]; ref = np.exp(np.mean(np.log(rv[gv])))
for h in range(4):
    cs = max(kd.C, key=lambda c: f1(Gv, np.full(len(rv), c), h, gv))                 # как активная голова: один порог на горизонт по val
    ss = max(np.exp(np.linspace(-2, 2, 41)), key=lambda s_: f1(Gv, rv / ref * s_, h, gv))
    out = []
    for part in ("val", "test"):
        G, r, g = res[part]; out.append((f1(G, np.full(len(r), cs), h, g), f1(G, r / ref * ss, h, g)))
    print(kd.HZ[h], "h  fixed c", round(cs, 3), "rel scale", round(ss, 3), " val fixed/rel %.3f/%.3f  test fixed/rel %.3f/%.3f" % (*out[0], *out[1]))
print("ref", ref)
