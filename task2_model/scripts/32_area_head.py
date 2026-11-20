"""Задание 2, шаг 32 (итерация 11). «Голова площади»: сколько км² сгорит за горизонт - отдельная модель по срезу,
и прогноз = столько клеток с наибольшей вероятностью U-Net (вместо единого порога вероятности).

Зачем: при едином пороге модель завышает площадь у мелких пожаров (×2–5) и занижает у крупных (> 50 км²: ×0,5–0,8, шаг 31).
Признаки среза: сводка карты U-Net (сумма вероятностей, клеток выше порога val, максимум), состояние пожара (площадь, термоточки,
возраст), погода окна GFS (ветер, T, RH, осадки), FWI GFWED, сезон и время суток.
Цель: log1p(новая гарь в сцене, км²). Модель: градиентный бустинг, обучение на val 2022 по 2 фолдам пожаров (cross-fit):
прогноз вне фолда -> подбор коэффициентов решения; на тест - среднее двух моделей.
Решение: n = round(alpha · Â / 0,25) клеток с наибольшей p среди клеток с p ≥ floor; alpha, floor - по val (вне фолда).
Вариант blend: n = (n_порог^(1-g) · n_площадь^g), g по val.
Выход: out_v3/area_head_<TAG>.csv (метрики), out_v3/area_head_<TAG>.joblib (модели + параметры), строки в size_diag.csv.
"""
import os, sys, importlib.util
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *


def _mod(name, f):
    s = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f)); m = importlib.util.module_from_spec(s)
    sys.modules[name] = m; s.loader.exec_module(m); return m


ev12 = _mod("ev12", "12_eval_all.py"); d31 = _mod("d31", "31_size_diag.py")
UNET = os.environ.get("UNET", "wx,wxbig,wxn,wxbign").split(","); TAG = os.environ.get("TAG", "all4")
d31.UNET = UNET; SRC = "gfsa"
CELL_KM2 = 0.25
SC = (["fc_gfsa_ws_mean_{H}h", "fc_gfsa_gust_max_{H}h", "fc_gfsa_T_max_{H}h", "fc_gfsa_RH_min_{H}h", "fc_gfsa_P_sum_{H}h"])
FIX = ["state_km2", "af_12h", "k_12h", "hour_utc", "gfwed_FWI", "gfwed_ISI", "gfwed_BUI", "gfwed_FFMC", "gfwed_DC",
       "era5_P_sum_prev7d", "era5_RH_min_prev24h", "era5_ws_mean_prev24h", "valid_frac", "lat", "lon"]


def slice_stats(P, Y, L, cal):
    """По срезу и горизонту: сумма p, n выше порога val, max p, факт (клеток); плюс сортированные p для решения."""
    n = len(P); F = np.zeros((len(HZ), n, 4), np.float32)
    for i in range(n):
        for h in range(len(HZ)):
            m = ev12.eval_mask(Y[i:i + 1], L[i:i + 1], h)[0]; p = P[i, h][m]
            F[h, i] = (p.astype(np.float32).sum() / 255, (p >= cal[h]["level"]).sum(), p.max() if p.size else 0, ((Y[i, h] > 0) & m).sum())
    return F


def design(si, F, h):
    H = HZ[h]
    X = np.column_stack([np.log1p(F[h, :, 0]), np.log1p(F[h, :, 1]), F[h, :, 2] / 255] +
                        [si[c.format(H=H)].astype(float).values for c in SC] +
                        [si[c].astype(float).values for c in FIX] +
                        [np.sin(2 * np.pi * pd.to_datetime(si.t_utc).dt.dayofyear / 365.25), np.cos(2 * np.pi * pd.to_datetime(si.t_utc).dt.dayofyear / 365.25)])
    X[:, 3 + len(SC)] = np.log1p(X[:, 3 + len(SC)]); X[:, 4 + len(SC)] = np.log1p(X[:, 4 + len(SC)])
    return X


def prep(P, Y, L, h):
    """Для быстрого решения: p по убыванию (вне маски -1) и накопленное число верных клеток."""
    n = len(P); Ps = np.empty((n, 96 * 96), np.int16); Cy = np.empty((n, 96 * 96), np.int32); npos = np.zeros(n)
    for i in range(n):
        m = ev12.eval_mask(Y[i:i + 1], L[i:i + 1], h)[0]; y = ((Y[i, h] > 0) & m).ravel()
        p = np.where(m, P[i, h].astype(np.int16), -1).ravel(); o = np.argsort(-p, kind="stable")
        Ps[i] = p[o]; Cy[i] = np.cumsum(y[o]); npos[i] = y.sum()
    return Ps, Cy, npos


def decide(D, n_cells, floor):
    """tp, fp, fn по срезам: top-n клеток по p среди p ≥ floor."""
    Ps, Cy, npos = D
    k = np.minimum(np.asarray(n_cells, np.int64), (Ps >= floor).sum(1)).astype(np.int64)
    tp = np.where(k > 0, Cy[np.arange(len(k)), np.maximum(k - 1, 0)], 0)
    return np.column_stack([tp, k - tp, npos - tp])


def f1(o):
    tp, fp, fn = o.sum(0); return 2 * tp / max(2 * tp + fp + fn, 1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sv, _, Yv, Lv = ev12.load_part("val"); Pv = d31.probs("val"); cal = ev12.fit_calib(Pv, Yv, Lv, rng)
    st, _, Yt, Lt = ev12.load_part("test"); Pt = d31.probs("test")
    Fv = slice_stats(Pv, Yv, Lv, cal); Ft = slice_stats(Pt, Yt, Lt, cal)
    fires = sv.fire_id.values; uf = np.unique(fires); rng.shuffle(uf); fold = np.isin(fires, uf[: len(uf) // 2]).astype(int)
    rows, size_rows, keep = [], [], {}
    for h, H in enumerate(HZ):
        Xv, Xt = design(sv, Fv, h), design(st, Ft, h); yv = np.log1p(Fv[h, :, 3] * CELL_KM2)
        oof = np.zeros(len(yv)); ms = []
        for f in (0, 1):
            mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                                                random_state=0).fit(Xv[fold != f], yv[fold != f])
            oof[fold == f] = mdl.predict(Xv[fold == f]); ms.append(mdl)
        At_hat = np.mean([m.predict(Xt) for m in ms], 0)
        nthr_v, nthr_t = Fv[h, :, 1], Ft[h, :, 1]; Dv, Dt = prep(Pv, Yv, Lv, h), prep(Pt, Yt, Lt, h)
        # подбор по val (вне фолда): alpha, floor, g
        best = (-1, None)
        for g in (0.0, 0.25, 0.5, 0.75, 1.0):
            for alpha in (0.6, 0.8, 1.0, 1.25, 1.5, 2.0):
                na = np.expm1(np.clip(oof, 0, None)) / CELL_KM2 * alpha
                n = np.round(np.maximum(nthr_v, 0.5) ** (1 - g) * np.maximum(na, 0.5) ** g * ((nthr_v > 0) | (na >= 1)))
                for floor in (cal[h]["level"] // 4, cal[h]["level"] // 2, cal[h]["level"]):
                    s = f1(decide(Dv, n, floor))
                    if s > best[0]: best = (s, (g, alpha, floor))
        g, alpha, floor = best[1]
        base_v = f1(decide(Dv, nthr_v, cal[h]["level"]))
        na = np.expm1(np.clip(At_hat, 0, None)) / CELL_KM2 * alpha
        nt = np.round(np.maximum(nthr_t, 0.5) ** (1 - g) * np.maximum(na, 0.5) ** g * ((nthr_t > 0) | (na >= 1)))
        Ot = decide(Dt, nt, floor); Ob = decide(Dt, nthr_t, cal[h]["level"])
        r = dict(model=TAG, h=H, g=g, alpha=alpha, floor=int(floor), val_base=base_v, val_head=best[0], test_base=f1(Ob), test_head=f1(Ot),
                 area_base=Ob[:, :2].sum() / max(Ob[:, [0, 2]].sum(), 1), area_head=Ot[:, :2].sum() / max(Ot[:, [0, 2]].sum(), 1),
                 corr_logA_val=float(np.corrcoef(oof, yv)[0, 1]))
        rows.append(r); print(r, flush=True); keep[H] = dict(models=ms, g=g, alpha=alpha, floor=int(floor), level=int(cal[h]["level"]))
        act = (Ot[:, 0] + Ot[:, 2]) * CELL_KM2; b = np.digitize(act, d31.BINS) - 1
        for k, nm in enumerate(d31.NAMES):
            m = b == k
            if not m.any(): continue
            tp, fp, fn = Ot[m].sum(0)
            size_rows.append(dict(model=TAG + "_areahead", part="test", h=H, size=nm, n=int(m.sum()), act_km2=float(act[m].sum()),
                                  F1=2 * tp / max(2 * tp + fp + fn, 1), recall=tp / max(tp + fn, 1), area_ratio=(tp + fp) / max(tp + fn, 1),
                                  pred0=float((Ot[m, :2].sum(1) == 0).mean())))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"area_head_{TAG}.csv"), index=False)
    joblib.dump(dict(features=dict(sc=SC, fix=FIX), horizons=keep), os.path.join(OUT, f"area_head_{TAG}.joblib"))
    f = os.path.join(OUT, "size_diag.csv"); d = pd.DataFrame(size_rows); o = pd.read_csv(f)
    pd.concat([o[o.model != TAG + "_areahead"], d]).to_csv(f, index=False)
    pd.set_option("display.width", 200); print(pd.DataFrame(rows).round(3).to_string(index=False)); print(d.round(3).to_string(index=False))
