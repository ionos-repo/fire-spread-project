"""ELMFIRE: пожар у Баянаула 5-6.10.2024 (6 очагов FIRMS) на четырёх картах топлива -> toa_<карта>.tif + viz_data.json.

  source ~/miniconda3/bin/activate elmfire; export PATH=$PATH:~/elmfire/build/linux/bin OMPI_MCA_osc=^ucx OMPI_MCA_pml=^ucx
  python elmfire_bench/bayanaul/run_bayanaul.py            (скачанное лежит в cache/)

Окно 60 x 60 км, сетка 100 м, UTM 43N. Рельеф - Copernicus DEM 30 м (AWS). Карты топлива FBFM40 (fuel/, вырезаны с сервера
gis, /data/arnai_fuelmapping/final_fuelampping), на сетку 100 м - модой:
  new  - новая карта Арная (output/fueltype_labeled_pt75e50n_k40.tif, 20 м, S2 + S1 + DEM, k-means k=40)
  old  - старая национальная (fire_spread/cases/bayanaul_20241006/fbfm40_old_pt75e50n.tif)
  elm  - карта настройки ELMFIRE на сервере (fbfm40_elmfire_pt75e50n.tif)
  wc   - ESA WorldCover 2021 -> FBFM40 таблицей task1 шаг 08 (наш датасет)
Погода - ERA5 (Open-Meteo, wx_era5.json) в одной точке, по часам: ветер 10 м -> 20 фт (x0,87); влажность мёртвого топлива
1 ч = EMC(T, RH), после дождя > 0,2 мм/ч +3 % на 3 ч; 10 ч / 100 ч - EMC, сглаженная за 12 / 48 ч, +1 / +2 %.
Живая трава в октябре сухая: 30 %. Зажигания - первая термоточка каждого очага (на 0,5 ч раньше: ELMFIRE завершает
расчёт, если в t = 0 ничего не горит). Скорость фронта как есть (adj = 1).
Факт - термоточки очагов (hotspots_2024-10-05_2024-10-06.geojson, 1276 точек): доля точек внутри расчётной гари к их времени.
"""
import json, os, shutil, subprocess
from datetime import datetime, timezone
import numpy as np, rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform as wtransform
from scipy.ndimage import binary_dilation

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
CRS, CELL, N = "EPSG:32643", 100.0, 600
LON0, LAT0 = 75.64, 50.13
T0 = datetime(2024, 10, 5, 20, 0, tzinfo=timezone.utc)                 # первая термоточка 20:27
TSTOP_H = 48
H_END = 25                                                             # последняя термоточка 06.10 20:55 UTC
WC2FB = {10: 161, 20: 121, 30: 102, 40: 101, 50: 91, 60: 99, 70: 92, 80: 98, 90: 104, 95: 98, 100: 99}
MAPS = dict(new="новая карта Арная", old="старая национальная", elm="карта ELMFIRE на сервере", wc="WorldCover (наш датасет)")
# очаг, точек, первая термоточка (UTC), lon, lat первой, последняя (UTC)
FOCI = [(418550, 35, "2024-10-05 20:27", 75.45641, 50.00335, "2024-10-06 08:20"),
        (418562, 16, "2024-10-06 04:46", 75.44550, 50.08769, "2024-10-06 08:43"),
        (418561, 259, "2024-10-06 04:46", 75.50906, 50.15074, "2024-10-06 08:43"),
        (418560, 415, "2024-10-06 04:46", 75.53529, 50.08058, "2024-10-06 08:43"),
        (418566, 429, "2024-10-06 06:16", 75.66198, 50.17675, "2024-10-06 20:55"),
        (418582, 122, "2024-10-06 07:56", 75.70511, 50.08047, "2024-10-06 15:56")]
hrs = lambda s: (datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc) - T0).total_seconds() / 3600

cx, cy = wtransform("EPSG:4326", CRS, [LON0], [LAT0]); cx, cy = round(cx[0] / CELL) * CELL, round(cy[0] / CELL) * CELL
TR = from_origin(cx - N / 2 * CELL, cy + N / 2 * CELL, CELL, CELL)
PROF = dict(width=N, height=N, crs=CRS, transform=TR)


def warp(urls, rs, nodata=None):
    out = np.full((N, N), np.nan, np.float32)
    for u in urls:
        with rasterio.open(u) as src:
            tmp = np.full((N, N), np.nan, np.float32)
            reproject(rasterio.band(src, 1), tmp, src_nodata=nodata if nodata is not None else src.nodata,
                      dst_transform=TR, dst_crs=CRS, resampling=rs, dst_nodata=np.nan)
        ok = np.isfinite(tmp) & ~np.isfinite(out); out[ok] = tmp[ok]
    return out


def cached(name, fn):
    p = os.path.join(CACHE, name + ".npy")
    if not os.path.exists(p):
        np.save(p, fn())
    return np.load(p)


def write(path, arr, dtype):
    arr = np.asarray(arr); arr = arr[None] if arr.ndim == 2 else arr
    with rasterio.open(path, "w", driver="GTiff", count=arr.shape[0], dtype=dtype, nodata=-9999, compress="deflate", **PROF) as r:
        r.write(arr.astype(dtype))


def emc(t_c, rh):
    t = t_c * 9 / 5 + 32
    return np.where(rh < 10, 0.03229 + 0.281073 * rh - 0.000578 * rh * t,
           np.where(rh < 50, 2.22749 + 0.160107 * rh - 0.01478 * t, 21.0606 + 0.005565 * rh ** 2 - 0.00035 * rh * t - 0.483199 * rh))


# ---------- ландшафт
DEM = [f"/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{t}_DEM/Copernicus_DSM_COG_10_{t}_DEM.tif"
       for t in ("N49_00_E075_00", "N50_00_E075_00", "N49_00_E076_00", "N50_00_E076_00")]
WC = ["/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N48E075_Map.tif"]
dem = cached("dem", lambda: warp(DEM, Resampling.bilinear))
gy, gx = np.gradient(dem, CELL)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
aspect = (np.degrees(np.arctan2(-gx, gy)) + 360) % 360
FUEL = {}
for k in MAPS:
    if k == "wc":
        lc = cached("worldcover", lambda: warp(WC, Resampling.mode))
        FUEL[k] = np.vectorize(lambda v: WC2FB.get(int(v), 99) if np.isfinite(v) else 99)(lc).astype(np.int16)
    else:
        a = warp([os.path.join(HERE, "fuel", f"fuel_{k}.tif")], Resampling.mode, nodata=255)
        a = np.where(np.isfinite(a), a, 99).astype(np.int16)
        a[(a < 91) | (a > 204)] = 99
        FUEL[k] = a

# ---------- погода по часам от T0
w = json.load(open(os.path.join(HERE, "wx_era5.json")))["hourly"]
tt = np.array([datetime.fromisoformat(s).replace(tzinfo=timezone.utc) for s in w["time"]])
T, RH, WS, WD, P = (np.array(w[k], float) for k in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "precipitation"))
e = emc(T, RH)
sm = lambda x, n: np.convolve(np.r_[np.full(n - 1, x[0]), x], np.ones(n) / n, "valid")
m1 = np.clip(e + 3 * np.convolve(P > 0.2, np.ones(3), "full")[:len(P)], 2, 60)
m10, m100 = np.clip(sm(e, 12) + 1, 3, 60), np.clip(sm(e, 48) + 2, 4, 60)
i0 = int(np.nonzero(tt == T0)[0][0]); sl = slice(i0, i0 + TSTOP_H + 1)
WX = dict(ws=WS[sl] * 0.87 * 2.23694, wd=WD[sl], m1=m1[sl], m10=m10[sl], m100=m100[sl],
          lh=np.full(TSTOP_H + 1, 30.0), lw=np.full(TSTOP_H + 1, 120.0))

# ---------- зажигания и термоточки
ign = []
for f in FOCI:
    x, y = wtransform("EPSG:4326", CRS, [f[3]], [f[4]])
    ign.append((x[0], y[0], max(hrs(f[2]) - 0.5, 0) * 3600))
ign_txt = "\n".join(f"X_IGN({i})={x:.1f}\nY_IGN({i})={y:.1f}\nT_IGN({i})={t:.1f}" for i, (x, y, t) in enumerate(ign, 1))
ids = {f[0] for f in FOCI}
hs = [h for h in json.load(open(os.path.join(HERE, "hotspots_2024-10-05_2024-10-06.geojson")))["features"] if h["properties"]["fireImageId"] in ids]
hx, hy = wtransform("EPSG:4326", CRS, [h["geometry"]["coordinates"][0] for h in hs], [h["geometry"]["coordinates"][1] for h in hs])
hc, hr = ~TR * (np.array(hx), np.array(hy))
ht = np.array([hrs(h["properties"]["date"] + " " + h["properties"]["time"][:5]) for h in hs])
hin = (hr >= 0) & (hr < N) & (hc >= 0) & (hc < N)
HR, HC = hr.astype(int).clip(0, N - 1), hc.astype(int).clip(0, N - 1)
seen = np.zeros((N, N), bool); seen[HR[hin], HC[hin]] = True
fact = binary_dilation(seen, iterations=4)                                # клетки в ~400 м от термоточки: грубый «факт»

DATA = open(os.path.join(HERE, "..", "run_elmfire.py"), encoding="utf-8").read().split('DATA = """')[1].split('"""')[0]
ELM = os.environ.get("ELMFIRE_BASE_DIR", os.path.expanduser("~/elmfire"))
VER = open(os.path.join(ELM, "VERSION")).read().strip()
BIN = os.environ.get("ELMFIRE_BIN") or shutil.which(f"elmfire_{VER}") or os.path.join(ELM, "build/linux/bin", f"elmfire_{VER}")

toa, stats = {}, {}
for k, fb in FUEL.items():
    run = os.path.join(HERE, f"run_{k}"); shutil.rmtree(run, ignore_errors=True)
    for s in ("inputs", "outputs", "scratch"):
        os.makedirs(os.path.join(run, s))
    z = np.zeros_like(fb)
    write(f"{run}/inputs/landscape.tif", np.stack([np.round(dem), np.round(np.clip(slope, 0, 89)), np.round(aspect) % 360, fb, z, z, z, z]), "int16")
    write(f"{run}/inputs/adj.tif", np.ones(fb.shape), "float32")
    write(f"{run}/inputs/phi.tif", np.ones(fb.shape), "float32")
    for q, v in WX.items():
        write(f"{run}/inputs/{q}.tif", np.broadcast_to(v[:, None, None], (len(v), N, N)), "float32")
    open(f"{run}/inputs/elmfire.data", "w").write(DATA.format(tstop=f"{TSTOP_H * 3600.0:.1f}", nmet=TSTOP_H + 1, nign=len(ign), ign=ign_txt))
    with open(f"{run}/elmfire.log", "w") as log:
        rc = subprocess.run([BIN, "./inputs/elmfire.data"], cwd=run, stdout=log, stderr=subprocess.STDOUT).returncode
    f = sorted(x for x in os.listdir(f"{run}/outputs") if x.startswith("time_of_arrival") and x.endswith((".bil", ".tif")))
    assert rc == 0 and f, f"ELMFIRE rc={rc}, см. {run}/elmfire.log"
    with rasterio.open(f"{run}/outputs/{f[0]}") as r:
        a = r.read(1).astype(np.float32); nod = r.nodata
    a[(a == nod) | ~np.isfinite(a) | (a <= 0)] = -1
    a = np.where(a > 0, a / 3600, -1).astype(np.float32)
    toa[k] = a; write(os.path.join(HERE, f"toa_{k}.tif"), a, "float32")
    b = a > 0
    at = a[HR[hin], HC[hin]]
    hit = (at > 0) & (at <= ht[hin] + 1)                                   # термоточка внутри гари к своему времени (+1 ч)
    v, c = np.unique(fb, return_counts=True)
    stats[k] = dict(name=MAPS[k], area_km2=[round(float(((a > 0) & (a <= h)).sum() * CELL ** 2 / 1e6), 1) for h in range(TSTOP_H + 1)],
                    hotspots_hit=round(float(hit.mean()), 3), hotspots_in_burn=round(float((at > 0).mean()), 3),
                    f1_fact=round(float(2 * (b & fact).sum() / (b.sum() + fact.sum())), 3),
                    fuel_pct={int(x): round(100 * y / fb.size, 1) for x, y in zip(v, c) if y / fb.size >= 0.005})
    b25 = (a > 0) & (a <= H_END)                                          # к последней термоточке (+25 ч)
    stats[k]["at_end"] = dict(h=H_END, area_km2=round(float(b25.sum() * CELL ** 2 / 1e6)), fact_km2=round(float(fact.sum() * CELL ** 2 / 1e6)),
                              hotspots_in=round(float(((at > 0) & (at <= H_END)).mean()), 2),
                              f1=round(float(2 * (b25 & fact).sum() / (b25.sum() + fact.sum())), 2),
                              precision=round(float((b25 & fact).sum() / max(b25.sum(), 1)), 2), recall=round(float((b25 & fact).sum() / fact.sum()), 2))
    print(k, MAPS[k], "| площадь 15/24/48 ч", [stats[k]["area_km2"][h] for h in (15, 24, 48)], "| точки в гари к сроку",
          stats[k]["hotspots_hit"], "| к 48 ч", stats[k]["hotspots_in_burn"], "| F1 с полосой у точек", stats[k]["f1_fact"], flush=True)

# ---------- данные для страницы (сетка 300 x 300 по 200 м)
k2 = 2
crn = wtransform(CRS, "EPSG:4326", [TR.c, TR.c + N * CELL, TR.c + N * CELL, TR.c], [TR.f, TR.f, TR.f - N * CELL, TR.f - N * CELL])
coarse_toa = lambda a: [int(v * 10) if np.isfinite(v) else -1 for v in np.where(a > 0, a, np.inf).reshape(N // k2, k2, N // k2, k2).min(axis=(1, 3)).ravel()]


def px(lon, lat):
    x, y = wtransform("EPSG:4326", CRS, [lon], [lat]); c, r = ~TR * (x[0], y[0])
    return [round(c / k2, 2), round(r / k2, 2)]


hours = [dict(h=h, ws=round(float(WS[i0 + h]), 1), wd=round(float(WD[i0 + h])), T=round(float(T[i0 + h]), 1),
              rh=round(float(RH[i0 + h])), p=round(float(P[i0 + h]), 1), m1=round(float(m1[i0 + h]), 1)) for h in range(TSTOP_H + 1)]
foci = [dict(id=f[0], n=f[1], first=f[2], last=f[5], t_first=round(hrs(f[2]), 2), ign=px(f[3], f[4]), lonlat=[f[3], f[4]]) for f in FOCI]
hot = [[round(float(c / k2), 1), round(float(r / k2), 1), round(float(t), 2)] for c, r, t, ok in zip(hc, hr, ht, hin) if ok]
json.dump(dict(t0=T0.isoformat(), n=N // k2, cell_m=CELL * k2, corners_lonlat=[list(p) for p in zip(*crn)],
               fuel={k: [int(v) for v in fb.reshape(N // k2, k2, N // k2, k2)[:, 0, :, 0].ravel()] for k, fb in FUEL.items()},
               toa={k: coarse_toa(a) for k, a in toa.items()}, maps=MAPS, stats=stats, hours=hours, foci=foci, hotspots=hot),
          open(os.path.join(HERE, "viz_data.json"), "w"), separators=(",", ":"))
json.dump(stats, open(os.path.join(HERE, "stats.json"), "w"), ensure_ascii=False, indent=1)
print("готово: viz_data.json", os.path.getsize(os.path.join(HERE, "viz_data.json")) // 1024, "КБ")
