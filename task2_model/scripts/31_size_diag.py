"""Задание 2, шаг 31. Диагностика по размеру роста: где модель теряет на крупных пожарах.
Ансамбль U-Net (UNET=wx,wxbig,wxn,wxbign, TTA, погода GFS) + калибровка/порог по val 2022 (как шаг 12),
затем на val и тесте F1, recall и отношение площадей по классам фактического роста за горизонт (км² новой гари в сцене).
Выход: out_v3/size_diag.csv"""
import os, sys, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *

s = importlib.util.spec_from_file_location("ev12", os.path.join(os.path.dirname(__file__), "12_eval_all.py")); ev12 = importlib.util.module_from_spec(s); s.loader.exec_module(ev12)
UNET = os.environ.get("UNET", "wx,wxbig,wxn,wxbign").split(","); SRC = os.environ.get("SRC", "gfsa"); TAG = os.environ.get("TAG", "all4")
BINS = [-0.01, 0.01, 1, 5, 20, 50, 1e9]; NAMES = ["0", "0-1", "1-5", "5-20", "20-50", ">50"]


def probs(part):
    P = None
    for u in UNET:
        p = np.load(os.path.join(OUT, f"probs_{u}_tta_{SRC}_{part}.npy"), mmap_mode="r").astype(np.float32)
        P = p if P is None else P + p
    return np.round(P / len(UNET)).astype(np.uint8)


def per_slice(P, Y, L, cal):
    n = len(P); out = np.zeros((len(HZ), 3, n))
    for i in range(n):
        for h in range(len(HZ)):
            m = ev12.eval_mask(Y[i:i + 1], L[i:i + 1], h)[0]; y = (Y[i, h] > 0) & m; pr = (P[i, h] >= cal[h]["level"]) & m
            out[h, :, i] = (pr & y).sum(), (pr & ~y).sum(), (~pr & y).sum()
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0); rows = []
    sv, _, Yv, Lv = ev12.load_part("val"); Pv = probs("val")
    cal = ev12.fit_calib(Pv, Yv, Lv, rng)
    for part in ("val", "test"):
        si, _, Y, L = ev12.load_part(part) if part != "val" else (sv, None, Yv, Lv)
        P = Pv if part == "val" else probs(part)
        S = per_slice(P, Y, L, cal)
        for h, H in enumerate(HZ):
            act = (S[h, 0] + S[h, 2]) * 0.25                                      # факт, км² (ячейка 0,25 км²)
            b = np.digitize(act, BINS) - 1
            for k, nm in enumerate(NAMES):
                m = b == k
                if not m.any(): continue
                tp, fp, fn = S[h][:, m].sum(1)
                rows.append(dict(model=TAG, part=part, h=H, size=nm, n=int(m.sum()), act_km2=float(act[m].sum()),
                                 F1=2 * tp / max(2 * tp + fp + fn, 1), recall=tp / max(tp + fn, 1),
                                 area_ratio=(tp + fp) / max(tp + fn, 1), pred0=float(((S[h, 0] + S[h, 1])[m] == 0).mean())))
    d = pd.DataFrame(rows); f = os.path.join(OUT, "size_diag.csv")
    if os.path.exists(f):
        o = pd.read_csv(f); d = pd.concat([o[o.model != TAG], d])
    d.to_csv(f, index=False)
    pd.set_option("display.width", 200)
    print(d[d.model == TAG].round(3).to_string(index=False))
