"""Задание 2, шаг 34 (итерация 13). Классификатор «взрывного роста» (Holden et al. 2025; Fire 2023, 6, 319).

По срезу: вероятность, что за горизонт h пожар даст новой гари в сцене больше порога (12/24 ч - 2 км², 48/72 ч - 5 км²),
и вероятность «роста нет» (0 клеток). Признаки - как у модели k (шаг 33: сцена у кромки, погода окон, FWI, сезон, место)
+ почва/VPD/HDW (KV=2). Обучение на train (ERA5), проверка AUC на val/тест (GFS). Выход для гибрида (шаг 26, BLOW=1):
out_v3/blow_<part>.npy [n, 2·4] - logit p(большой рост) и logit p(нет роста) по 4 горизонтам; out_v3/blowup.joblib.
Признаки заповедников считаются здесь же (кэш out_v3/k_dyn_cache_reserves.npz) - это дольше всего, переживает перезапуск.
"""
import os, sys, importlib.util
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *

_s = importlib.util.spec_from_file_location("kd", os.path.join(os.path.dirname(__file__), "33_k_dyn.py"))
kd = importlib.util.module_from_spec(_s); _s.loader.exec_module(kd)
ev12 = kd.ev12
BIG = np.array([2, 2, 5, 5]) / 0.25                                   # клеток 0,25 км²


def act(Y, L):
    a = np.zeros((len(Y), len(HZ)), np.int32)
    for i in range(len(Y)):
        for h in range(len(HZ)):
            m = ev12.eval_mask(Y[i:i + 1], L[i:i + 1], h)[0]; a[i, h] = ((Y[i, h] > 0) & m).sum()
    return a


def load(part, src):
    f = os.path.join(OUT, f"k_dyn_cache_{part}.npz"); si, M, Y, L = ev12.load_part(part)
    if os.path.exists(f):
        z = np.load(f); X, A = z["X"], z["ACT"]
        if "ACT" in z and len(X) == len(si):
            return si, np.column_stack([X, kd.extra(si, src)]), A
        n = len(X); si = si.iloc[:n].reset_index(drop=True); return si, np.column_stack([X, kd.extra(si, src)]), A
    X = kd.features(M, si, src); A = act(Y, L); np.savez(f, X=X, ACT=A)
    return si, np.column_stack([X, kd.extra(si, src)]), A


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p)).astype(np.float32)


if __name__ == "__main__":
    D = {p: load(p, s) for p, s in (("train", "era5"), ("val", "gfsa"), ("test", "gfsa"), ("reserves", "era5"))}
    for p in D: print(p, len(D[p][0]), "ready", flush=True)
    _, Xt, At = D["train"]; models = {}
    for h in range(len(HZ)):
        for kind, y in (("big", At[:, h] >= BIG[h]), ("zero", At[:, h] == 0)):
            models[(kind, h)] = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                                                               l2_regularization=1.0, random_state=0).fit(Xt, y)
    rows = []
    for p in ("val", "test", "reserves"):
        si, X, A = D[p]; out = []
        for kind in ("big", "zero"):
            for h in range(len(HZ)):
                q = models[(kind, h)].predict_proba(X)[:, 1]; out.append(lg(q))
                y = (A[:, h] >= BIG[h]) if kind == "big" else (A[:, h] == 0)
                if 0 < y.sum() < len(y):
                    rows.append(dict(part=p, kind=kind, h=HZ[h], pos=round(float(y.mean()), 3), auc=roc_auc_score(y, q), ap=average_precision_score(y, q)))
        np.save(os.path.join(OUT, f"blow_{p}.npy"), np.column_stack(out))
    joblib.dump(dict(models=models, big=BIG.tolist(), maps=kd.MAPS, scal=kd.SCAL, wx=kd.WX, extra=kd.EXTRA_S), os.path.join(OUT, "blowup.joblib"))
    d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "blowup.csv"), index=False); print(d.round(3).to_string(index=False))
