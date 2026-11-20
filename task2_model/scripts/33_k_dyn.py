"""Задание 2, шаг 33 (итерация 12). Переменный множитель скорости k_ros вместо одного 0,05 на всю страну.

Время прихода по физике T посчитано при k0 = 0,05 (шаги 19/27). При k = c·k0 скорости ×c, поэтому прогноз на горизонт h ≈
{T ≤ c·h} (точно - при постоянном ветре; при смене ветра по окнам - приближение). Для каждого среза по сетке c считаются
площадь и верные клетки прогноза физики - без пересчёта физики.

1) k по местности и сезону («модель k»): на train для срезов с ростом за 72 ч ≥ 1 км² - c*, при котором площадь физики
   ближе всего к факту (по 4 горизонтам, в логарифме). Градиентный бустинг log c* по признакам сцены у кромки пожара
   (доли топлива, FBFM40, NDVI, аномалия, рельеф) и погоде/FWI/сезону/месту.
2) Подстройка по ходу пожара («онлайн»): у среза t есть срез того же пожара t−24 ч; его рост за 24 ч к моменту t уже
   наблюдён - c_on подбирается по нему. Итог: log c = w·log c_on + (1−w)·log c_модели (w по val), без среза t−24 ч - модель.
Оценка: только физика (как шаг 14), val и тест (GFS), площадь и F1; варианты: k0 фикс., k по val (один), модель, модель+онлайн.
Выход: out_v3/k_dyn.csv, out_v3/k_dyn.joblib, out_v3/k_dyn_c_<part>.npy (c по срезам, для гибрида и прода).
"""
import os, sys, importlib.util
import numpy as np, pandas as pd, joblib
from scipy.ndimage import binary_dilation
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *

_s = importlib.util.spec_from_file_location("ev12", os.path.join(os.path.dirname(__file__), "12_eval_all.py"))
ev12 = importlib.util.module_from_spec(_s); _s.loader.exec_module(ev12)
C = np.exp(np.linspace(np.log(0.05), np.log(20), 49))               # сетка множителя c (k = c · 0,05)
K0 = 0.05
MAPS = ["grass_frac", "crop_frac", "shrub_tree_frac", "nonburnable_frac", "ndvi", "ndvi_anom", "ndvi_peak_prev", "slope_deg",
        "elev", "prior_burn", "road_density"]
SCAL = ["gfwed_FWI", "gfwed_ISI", "gfwed_BUI", "gfwed_FFMC", "gfwed_DC", "era5_P_sum_prev7d", "era5_RH_min_prev24h", "state_km2",
        "af_12h", "k_12h", "hour_utc", "lat", "lon"]
WX = ["ws_mean", "gust_max", "T_max", "RH_min", "P_sum"]
KV = os.environ.get("KV", "1")                                     # it13: KV=2 - + влажность почвы, VPD, HDW, DMC, FWI_max3d (Holden 2025)
SFX = ("" if KV == "1" else f"_v{KV}") + ("_mono" if os.environ.get("MONO") else "")
# it13: MONO=1 - log c не убывает с ветром (ws_mean, gust_max окон 24/72 ч): площадь физики×c растёт с ветром (проверка prod/checks 2)
EXTRA_S = ["era5_soilm_t", "era5_VPD_mean_prev7d", "gfwed_DMC", "gfwed_FWI_max3d"]


def vpd(T, RH):
    """кПа по T °C и RH % (Тетенс)."""
    return 0.6108 * np.exp(17.27 * T / (T + 237.3)) * (1 - np.clip(RH, 0, 100) / 100)


def extra(si, src):
    """it13: доп. признаки из индекса срезов (без карт): почва, VPD, HDW = ветер × VPD (прогнозные окна 24/72 ч)."""
    pre = "fc_gfsa" if src == "gfsa" else "era5"; g = lambda c: si[c].astype(float).values
    cols = [g(c) for c in EXTRA_S]
    for H in (24, 72):
        v = vpd(g(f"{pre}_T_max_{H}h"), g(f"{pre}_RH_min_{H}h")); cols += [v, v * g(f"{pre}_ws_mean_{H}h"), v * g(f"{pre}_gust_max_{H}h")]
    return np.column_stack(cols)


EXTRA_N = EXTRA_S + [f"{n}_{H}h" for H in (24, 72) for n in ("VPD", "HDW", "HDWg")]


MONO_IDX = {len(MAPS) + k * len(WX) + WX.index(w) for k in range(2) for w in ("ws_mean", "gust_max")}


def grid(T, Y, L):
    """По срезу и горизонту для каждого c: клеток прогноза и верных клеток; факт."""
    n = len(T); A = np.zeros((n, len(HZ), len(C)), np.int32); TP = np.zeros_like(A); ACT = np.zeros((n, len(HZ)), np.int32)
    for i in range(n):
        t = np.asarray(T[i], np.float32)
        for h, H in enumerate(HZ):
            m = ev12.eval_mask(Y[i:i + 1], L[i:i + 1], h)[0]; y = (Y[i, h] > 0) & m
            tt = np.where(m, np.nan_to_num(t, nan=1e9, posinf=1e9), 1e9).ravel(); o = np.argsort(tt); ts = tt[o]; cy = np.concatenate([[0], np.cumsum(y.ravel()[o])])
            k = np.searchsorted(ts, C * H, side="right"); A[i, h] = k; TP[i, h] = cy[k]; ACT[i, h] = y.sum()
    return A, TP, ACT


def features(M, si, src):
    """Сцена у кромки: среднее карт кэша (nn_maps, каналы TRANS) в кольце 1–5 клеток вокруг гари; + скаляры и погода окна 24/72 ч."""
    TR = list(TRANS); ib = TR.index("burned_frac"); X = np.zeros((len(si), len(MAPS)), np.float32)
    for i in range(len(si)):
        b = np.asarray(M[i, ib]) > 0
        ring = binary_dilation(b, iterations=5) & ~b if b.any() else np.ones((96, 96), bool)
        for j, nm in enumerate(MAPS):
            v = np.asarray(M[i, TR.index(nm)], np.float32)[ring]; X[i, j] = np.nanmean(v) if v.size else np.nan
    pre = "fc_gfsa" if src == "gfsa" else "era5"
    W = np.column_stack([si[f"{pre}_{w}_{H}h"].astype(float).values for H in (24, 72) for w in WX])
    S = si[SCAL].astype(float).values.copy(); S[:, SCAL.index("state_km2")] = np.log1p(S[:, SCAL.index("state_km2")]); S[:, SCAL.index("af_12h")] = np.log1p(S[:, SCAL.index("af_12h")])
    d = pd.to_datetime(si.t_utc).dt.dayofyear.values
    return np.column_stack([X, W, S, np.sin(2 * np.pi * d / 365.25), np.cos(2 * np.pi * d / 365.25)])


def c_star(A, ACT, hs=(0, 1, 2, 3)):
    err = sum((np.log1p(A[:, h]) - np.log1p(ACT[:, h, None])) ** 2 for h in hs)
    return C[np.argmin(err, 1)]


def at_c(G, c):
    """Метрики при c по срезам (вектор): tp, fp, fn по горизонтам."""
    A, TP, ACT = G; j = np.clip(np.searchsorted(C, c), 0, len(C) - 1); ii = np.arange(len(j))
    return [(TP[ii, h, j], A[ii, h, j] - TP[ii, h, j], ACT[:, h] - TP[ii, h, j]) for h in range(len(HZ))]


def summ(G, c, tag, part):
    rows = []
    for h, (tp, fp, fn) in enumerate(at_c(G, c)):
        T_, F_, N_ = tp.sum(), fp.sum(), fn.sum(); big = (tp + fn) * 0.25 >= 20
        rows.append(dict(variant=tag, part=part, h=HZ[h], F1=2 * T_ / max(2 * T_ + F_ + N_, 1), area_ratio=(T_ + F_) / max(T_ + N_, 1),
                         area_ratio_big20=(tp[big] + fp[big]).sum() / max((tp[big] + fn[big]).sum(), 1),
                         med_log_area_err=float(np.median(np.abs(np.log1p(tp + fp) - np.log1p(tp + fn))[(tp + fn) > 0]))))
    return rows


def online(si, G):
    """c_on для среза t по срезу того же пожара t−24 ч (его рост за 24 ч уже наблюдён); нет - nan."""
    A, _, ACT = G; key = {(f, t): i for i, (f, t) in enumerate(zip(si.fire_id.values, pd.to_datetime(si.t_utc).values))}
    c = np.full(len(si), np.nan)
    for i, (f, t) in enumerate(zip(si.fire_id.values, pd.to_datetime(si.t_utc).values)):
        j = key.get((f, t - np.timedelta64(24, "h")))
        if j is not None and ACT[j, 1] >= 4:                  # рост ≥ 1 км² за прошлые сутки
            c[i] = C[np.argmin(np.abs(np.log1p(A[j, 1]) - np.log1p(ACT[j, 1])))]
    return c


if __name__ == "__main__":
    data = {}
    for part, src in (("train", "era5"), ("val", "gfsa"), ("test", "gfsa")):
        si, M, Y, L = ev12.load_part(part); T = np.load(os.path.join(OUT, f"phys_T_{part}_{src}.npy"), mmap_mode="r")
        n = min(len(si), len(T)); si = si.iloc[:n].reset_index(drop=True)
        f = os.path.join(OUT, f"k_dyn_cache_{part}.npz")
        if os.path.exists(f):
            z = np.load(f); G, X = (z["A"], z["TP"], z["ACT"]), z["X"]
        else:
            G = grid(T[:n], Y, L); X = features(M, si, src); np.savez(f, A=G[0], TP=G[1], ACT=G[2], X=X)
        if KV != "1":
            X = np.column_stack([X, extra(si, src)])
        data[part] = (si, G, X); print(part, n, "ready", flush=True)

    si, G, X = data["train"]; grow = G[2][:, 3] >= 4                # рост за 72 ч ≥ 1 км²
    cs = c_star(G[0], G[2]); y = np.log(cs)
    mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0,
                                        random_state=0,
                                        monotonic_cst=[1 if j in MONO_IDX else 0 for j in range(X.shape[1])] if os.environ.get("MONO") else None).fit(X[grow], y[grow])
    print("train c*: median", np.median(cs[grow]).round(3), "IQR", np.percentile(cs[grow], [25, 75]).round(3), flush=True)

    sv, Gv, Xv = data["val"]; cv_model = np.exp(mdl.predict(Xv))
    # один k по val (как сейчас, но с перекалибровкой) и сдвиг модели по val
    f1v = lambda c: np.mean([2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1) for tp, fp, fn in at_c(Gv, c)])
    c_one = max(C, key=lambda c: f1v(np.full(len(sv), c)))
    s_best = max(np.exp(np.linspace(-1.5, 1.5, 31)), key=lambda s: f1v(cv_model * s))
    con_v = online(sv, Gv)
    w_best = max(np.linspace(0, 1, 11), key=lambda w: f1v(np.where(np.isnan(con_v), cv_model * s_best, np.exp(w * np.log(np.nan_to_num(con_v, nan=1)) + (1 - w) * np.log(cv_model * s_best)))))
    print(f"val: один k = {c_one * K0:.4f}, сдвиг модели ×{s_best:.2f}, вес онлайн {w_best:.1f}, доля срезов с онлайн {np.mean(~np.isnan(con_v)):.2f}", flush=True)

    rows = []
    for part in ("val", "test"):
        s_, G_, X_ = data[part]; cm = np.exp(mdl.predict(X_)) * s_best; con = online(s_, G_)
        cb = np.where(np.isnan(con), cm, np.exp(w_best * np.log(np.nan_to_num(con, nan=1)) + (1 - w_best) * np.log(cm)))
        np.save(os.path.join(OUT, f"k_dyn_c_{part}{SFX}.npy"), np.column_stack([cm, cb]).astype(np.float32))
        for tag, c in (("k0=0.05", np.ones(len(s_))), (f"один k={c_one * K0:.3f}", np.full(len(s_), c_one)), ("модель k", cm), ("модель k + онлайн", cb),
                       ("оракул c*", c_star(G_[0], G_[2]))):
            rows += summ(G_, c, tag, part)
    d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, f"k_dyn{SFX}.csv"), index=False)
    joblib.dump(dict(model=mdl, scale=float(s_best), w_online=float(w_best), k0=K0, maps=MAPS, scal=SCAL, wx=WX, c_one=float(c_one), kv=KV, mono=bool(os.environ.get("MONO")), extra=EXTRA_S if KV != "1" else []), os.path.join(OUT, f"k_dyn{SFX}.joblib"))
    pd.set_option("display.width", 220); print(d.round(3).to_string(index=False))
    imp = pd.Series(np.abs(np.corrcoef(np.column_stack([X[grow], y[grow]]).T)[-1, :-1]), index=MAPS + [f"{w}_{H}h" for H in (24, 72) for w in WX] + SCAL + ["doy_s", "doy_c"] + (EXTRA_N if KV != "1" else []))
    print("|corr| с log c*:", imp.sort_values(ascending=False).head(10).round(2).to_dict())
