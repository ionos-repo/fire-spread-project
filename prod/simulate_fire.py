"""Симуляция одного пожара со своей картой топлива (карта может покрывать только район пожара).

    python -X utf8 -m prod.simulate_fire --lon 80.3 --lat 50.3 --t "2024-08-10 06:00" --hours 72 --fuel_tif my_fuel.tif --out fire1
    python -X utf8 -m prod.simulate_fire ... --fuel_tif my_fuel.tif --compare      # + тот же прогон на базовом топливе

--t        время возгорания, UTC. Погода реальная: архив ERA5 / прогноз GFS через open-meteo (нужен интернет).
           Без интернета или для «что если» - задать погоду вручную: --wind 8 270 [--T 30 --RH 20]  (м/с, откуда дует, град).
--fuel_tif GeoTIFF FBFM40, EPSG:4326, любой охват (только район пожара - нормально): вне карты топливо базовое.
           Переводится на сетку модели один раз, результат кэшируется рядом (<tif>.lc500.npz).
--mode     active (по умолчанию, физика по времени прихода огня) | expected (гибрид, вероятностный прогноз).
Выход:  <out>_<fuel>.geojson - контуры гари каждые 12 ч (свойства hour, area_km2), <out>_<fuel>.png - картинка, сводка в консоли.
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--lon", type=float, required=True); ap.add_argument("--lat", type=float, required=True)
ap.add_argument("--t", required=True); ap.add_argument("--hours", type=int, default=72)
ap.add_argument("--radius_km", type=float, default=0.5); ap.add_argument("--mode", default="active", choices=["active", "expected"])
ap.add_argument("--fuel_tif"); ap.add_argument("--compare", action="store_true")
ap.add_argument("--wind", type=float, nargs=2, metavar=("WS", "WD"))
ap.add_argument("--T", type=float, default=30.0); ap.add_argument("--RH", type=float, default=20.0)
ap.add_argument("--out", default="fire")
a = ap.parse_args()

from .config import KZ                                                      # noqa: E402

runs = []
if a.fuel_tif:
    lc = os.path.splitext(a.fuel_tif)[0] + ".lc500.npz"
    if not os.path.exists(lc) or os.path.getmtime(lc) < os.path.getmtime(a.fuel_tif):
        from . import fuel_import
        fuel_import.main(["--tif", a.fuel_tif, "--base", os.path.join(KZ, "lc.npz"),
                          "--meta", os.path.join(KZ, "slices_v3_channels.json"), "--out", lc])
    runs.append(("custom", lc))
if a.compare or not a.fuel_tif:
    runs.append(("base", None))

import numpy as np                                                          # noqa: E402
from .engine import FireEngine                                              # noqa: E402

weather = "gfs" if a.wind is None else dict(winds=tuple(a.wind), T=a.T, RH=a.RH, use_real_history=False)
res = {}
for tag, lc in runs:
    if lc: os.environ["FIRE_LC"] = lc
    else: os.environ.pop("FIRE_LC", None)
    E = FireEngine()
    r, c = E.land.rc(a.lon, a.lat)
    fr = E.land.FR[:, max(r - 5, 0):r + 6, max(c - 5, 0):c + 6].mean((1, 2))
    print(f"\n[{tag}] топливо вокруг очага (5 км), %: трава {fr[0]:.0f}, пашня {fr[1]:.0f}, кусты/лес {fr[2]:.0f}, негорючее {fr[4]:.0f}")
    s = E.simulate(a.lon, a.lat, t_utc=a.t, hours=a.hours, radius_km=a.radius_km, weather=weather, mode=a.mode)
    feats = []
    for f in s["frames"]:
        g = f["geojson"]
        for ft in (g if isinstance(g, list) else g.get("features", [])):
            ft = dict(ft); ft["properties"] = dict(ft.get("properties") or {}, hour=f["hour"], area_km2=round(f["area_km2"], 2), fuel=tag)
            feats.append(ft)
        print(f"  {f['hour']:5.0f} ч: {f['area_km2']:8.1f} км2" + ("   (упёрлось в край окна)" if f.get("edge_clipped") else ""))
    fn = f"{a.out}_{tag}.geojson"
    json.dump(dict(type="FeatureCollection", features=feats), open(fn, "w", encoding="utf8"), ensure_ascii=False)
    res[tag] = s; print("  ->", fn)

    try:                                                                    # картинка: время прихода огня
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ks = np.array([[int(v) for v in k.split(",")] for k in s["arrival_h"]]); T = np.array(list(s["arrival_h"].values()))
        img = np.full((np.ptp(ks[:, 0]) + 21, np.ptp(ks[:, 1]) + 21), np.nan); img[ks[:, 0] - ks[:, 0].min() + 10, ks[:, 1] - ks[:, 1].min() + 10] = T
        bg = E.land.FR[0, ks[:, 0].min() - 10:ks[:, 0].min() - 10 + img.shape[0], ks[:, 1].min() - 10:ks[:, 1].min() - 10 + img.shape[1]]
        fig, ax = plt.subplots(figsize=(7, 6)); ax.imshow(bg, cmap="Greens", vmin=0, vmax=100, alpha=0.5)
        im = ax.imshow(img, cmap="inferno_r"); plt.colorbar(im, label="приход огня, ч"); ax.set_title(f"{tag}: {a.lon}, {a.lat}, {a.t}  (ячейка 500 м, фон - доля травы)")
        fig.savefig(f"{a.out}_{tag}.png", dpi=110, bbox_inches="tight"); plt.close(fig); print("  ->", f"{a.out}_{tag}.png")
    except Exception as e:
        print("  png не сделан:", e)

if len(res) == 2:
    A = {f["hour"]: f["area_km2"] for f in res["custom"]["frames"]}; B = {f["hour"]: f["area_km2"] for f in res["base"]["frames"]}
    print("\nчас   своё топливо   базовое")
    for h in sorted(A): print(f"{h:4.0f}   {A[h]:10.1f}   {B.get(h, float('nan')):9.1f}")
