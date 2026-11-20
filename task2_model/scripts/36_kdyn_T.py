"""Задание 2, шаг 36 (it13). Время прихода с переменным k: phys_T_<part>_<src>_kdyn<SFX>.npy = T(k0) / c (c - модель k шага 33).
SFX - суффикс модели (KSFX=_mono -> k_dyn_mono.joblib). Признаки - кэш out_v3/k_dyn_cache_<part>.npz (заповедники - из шага 34).
Пропускает готовые файлы (возобновляемо)."""
import os, sys, numpy as np, joblib
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import OUT
S = os.environ.get("KSFX", "_mono"); k = joblib.load(os.path.join(OUT, f"k_dyn{S}.joblib"))
for part, src in (("val", "gfsa"), ("test", "gfsa"), ("reserves", "era5")):
    fo = os.path.join(OUT, f"phys_T_{part}_{src}_kdyn{S}.npy")
    if os.path.exists(fo):
        print("skip", fo); continue
    X = np.load(os.path.join(OUT, f"k_dyn_cache_{part}.npz"))["X"]; c = (np.exp(k["model"].predict(X)) * k["scale"]).astype(np.float32)
    T = np.load(os.path.join(OUT, f"phys_T_{part}_{src}.npy"), mmap_mode="r"); n = min(len(T), len(c))
    out = np.lib.format.open_memmap(fo + ".tmp.npy", "w+", T.dtype, (n,) + T.shape[1:])
    for i in range(0, n, 512):
        out[i:i + 512] = (np.asarray(T[i:i + 512], np.float32) / c[i:i + 512, None, None]).astype(T.dtype)
    out.flush(); del out; os.replace(fo + ".tmp.npy", fo); np.save(os.path.join(OUT, f"k_dyn_c_{part}{S}_cm.npy"), c)
    print(part, n, "c median", np.median(c).round(3), flush=True)
