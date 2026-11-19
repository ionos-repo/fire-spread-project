"""ELMFIRE: пожар у Баянаула 5-6.10.2026 (6 очагов по первым термоточкам FIRMS) -> outputs/toa_adj<k>.tif + viz_data.json.

  source ~/miniconda3/bin/activate elmfire; export PATH=$PATH:~/elmfire/build/linux/bin OMPI_MCA_osc=^ucx OMPI_MCA_pml=^ucx
  python elmfire_bench/bayanaul/run_bayanaul.py            (повторный запуск берёт скачанное из cache/)

Окно 60 x 60 км, сетка 100 м, UTM 43N. Рельеф - Copernicus DEM 30 м (AWS), топливо - ESA WorldCover 2021 -> FBFM40
(та же таблица, что в task1 шаг 08). Погода - ERA5 (Open-Meteo, wx_era5.json) в одной точке, по часам:
ветер 10 м -> 20 фт (x0,87), влажность мёртвого топлива 1 ч = EMC(T, RH) по часам, 10 ч / 100 ч - EMC, сглаженная
за 12 / 48 ч, +1 / +2 %; дождь > 0,2 мм/ч добавляет к 1 ч топливу. Живая трава в октябре сухая: 30 %.
Зажигания - первая термоточка каждого очага, в её время. Два прогона: ELMFIRE как есть (adj 1) и с поправкой
скорости x0,05 (как k_ros нашей FBP).
"""
import json, os, shutil, subprocess
from datetime import datetime, timezone
import numpy as np, rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform as wtransform

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
CRS, CELL, N = "EPSG:32643", 100.0, 600
LON0, LAT0 = 75.64, 50.13
T0 = datetime(2026, 10, 5, 20, 0, tzinfo=timezone.utc)                 # первая термоточка 20:27 -> зажигание в 20:00
TSTOP_H = 48
FBFM40 = {10: 161, 20: 121, 30: 102, 40: 101, 50: 91, 60: 99, 70: 92, 80: 98, 90: 104, 95: 98, 100: 99}
# очаг, точек, первая термоточка (UTC), lon, lat первой, охват всех точек lon0, lon1, lat0, lat1, последняя (UTC)
FOCI = [(418550, 35, "2026-10-05 20:27", 75.45641, 50.00335, 75.4282, 75.5350, 49.9966, 50.0251, "2026-10-06 08:20"),
        (418562, 16, "2026-10-06 04:46", 75.44550, 50.08769, 75.4312, 75.4642, 50.0685, 50.1209, "2026-10-06 08:43"),
        (418561, 259, "2026-10-06 04:46", 75.50906, 50.15074, 75.4675, 75.7536, 50.1047, 50.2589, "2026-10-06 08:43"),
        (418560, 415, "2026-10-06 04:46", 75.53529, 50.08058, 75.5206, 75.7467, 50.0216, 50.1504, "2026-10-06 08:43"),
        (418566, 429, "2026-10-06 06:16", 75.66198, 50.17675, 75.6383, 75.7845, 50.0981, 50.2611, "2026-10-06 20:55"),
        (418582, 122, "2026-10-06 07:56", 75.70511, 50.08047, 75.7051, 75.8551, 50.0474, 50.1396, "2026-10-06 15:56")]
hrs = lambda s: (datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc) - T0).total_seconds() / 3600

cx, cy = wtransform("EPSG:4326", CRS, [LON0], [LAT0]); cx, cy = round(cx[0] / CELL) * CELL, round(cy[0] / CELL) * CELL
TR = from_origin(cx - N / 2 * CELL, cy + N / 2 * CELL, CELL, CELL)
PROF = dict(width=N, height=N, crs=CRS, transform=TR)


def warp(urls, rs, dtype):
    out = np.zeros((N, N), np.float32); filled = np.zeros((N, N), bool)
    for u in urls:
        with rasterio.open(u) as src:
            tmp = np.full((N, N), np.nan, np.float32)
            reproject(rasterio.band(src, 1), tmp, dst_transform=TR, dst_crs=CRS, resampling=rs, dst_nodata=np.nan)
        ok = np.isfinite(tmp) & ~filled; out[ok] = tmp[ok]; filled |= ok
    return out.astype(dtype), filled


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
dem = cached("dem", lambda: warp(DEM, Resampling.bilinear, np.float32)[0])
lc = cached("worldcover", lambda: warp(WC, Resampling.mode, np.float32)[0]).astype(np.int32)
fb = np.vectorize(lambda v: FBFM40.get(int(v), 99))(lc).astype(np.int16)
gy, gx = np.gradient(dem, CELL)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
aspect = (np.degrees(np.arctan2(-gx, gy)) + 360) % 360

# ---------- погода по часам от T0
w = json.load(open(os.path.join(HERE, "wx_era5.json")))["hourly"]
tt = np.array([datetime.fromisoformat(s).replace(tzinfo=timezone.utc) for s in w["time"]])
T, RH, WS, WD, P = (np.array(w[k], float) for k in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "precipitation"))
e = emc(T, RH)
sm = lambda x, n: np.convolve(np.r_[np.full(n - 1, x[0]), x], np.ones(n) / n, "valid")
m1 = np.clip(e + 3 * np.convolve(P > 0.2, np.ones(3), "full")[:len(P)], 2, 60)    # мокрое топливо ещё ~3 ч после дождя
m10, m100 = np.clip(sm(e, 12) + 1, 3, 60), np.clip(sm(e, 48) + 2, 4, 60)
i0 = int(np.nonzero(tt == T0)[0][0]); sl = slice(i0, i0 + TSTOP_H + 1)
WX = dict(ws=WS[sl] * 0.87 * 2.23694, wd=WD[sl], m1=m1[sl], m10=m10[sl], m100=m100[sl],
          lh=np.full(TSTOP_H + 1, 30.0), lw=np.full(TSTOP_H + 1, 120.0))

# ---------- зажигания
ign = []
for f in FOCI:
    x, y = wtransform("EPSG:4326", CRS, [f[3]], [f[4]])
    ign.append((x[0], y[0], max(hrs(f[2]) - 0.5, 0) * 3600))   # -0,5 ч: первый очаг в t = 0, иначе ELMFIRE завершает расчёт без горящих клеток
ign_txt = "\n".join(f"X_IGN({i})={x:.1f}\nY_IGN({i})={y:.1f}\nT_IGN({i})={t:.1f}" for i, (x, y, t) in enumerate(ign, 1))

DATA = open(os.path.join(HERE, "..", "run_elmfire.py"), encoding="utf-8").read().split('DATA = """')[1].split('"""')[0]
ELM = os.environ.get("ELMFIRE_BASE_DIR", os.path.expanduser("~/elmfire"))
VER = open(os.path.join(ELM, "VERSION")).read().strip()
BIN = os.environ.get("ELMFIRE_BIN") or shutil.which(f"elmfire_{VER}") or os.path.join(ELM, "build/linux/bin", f"elmfire_{VER}")

toa = {}
for adj in (1.0, 0.05):
    run = os.path.join(HERE, f"run_adj{adj:g}"); shutil.rmtree(run, ignore_errors=True)
    for s in ("inputs", "outputs", "scratch"):
        os.makedirs(os.path.join(run, s))
    z = np.zeros_like(fb)
    write(f"{run}/inputs/landscape.tif", np.stack([np.round(dem), np.round(np.clip(slope, 0, 89)), np.round(aspect) % 360, fb, z, z, z, z]), "int16")
    write(f"{run}/inputs/adj.tif", np.full(fb.shape, adj), "float32")
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
    a[(a == nod) | ~np.isfinite(a) | (a < 0)] = -1
    toa[adj] = a / 3600 * (a > 0) + (a <= 0) * -1
    write(os.path.join(HERE, f"toa_adj{adj:g}.tif"), toa[adj], "float32")
    print(f"adj {adj:g}: сгорело за {TSTOP_H} ч {(toa[adj] > 0).sum() * CELL ** 2 / 1e6:.1f} км²", flush=True)

# ---------- данные для страницы (сетка 300 x 300 по 200 м, lon/lat углов окна)
k = 2
crn = wtransform(CRS, "EPSG:4326", [TR.c, TR.c + N * CELL, TR.c + N * CELL, TR.c], [TR.f, TR.f, TR.f - N * CELL, TR.f - N * CELL])


def coarse_toa(a):
    b = np.where(a > 0, a, np.inf).reshape(N // k, k, N // k, k).min(axis=(1, 3))
    return [int(v * 10) if np.isfinite(v) else -1 for v in b.ravel()]          # десятые часа


def lonlat_to_px(lon, lat):
    x, y = wtransform("EPSG:4326", CRS, [lon], [lat]); c, r = ~TR * (x[0], y[0])
    return round(c / k, 2), round(r / k, 2)


fbc = fb.reshape(N // k, k, N // k, k)[:, 0, :, 0]
hours = [dict(h=h, t=(T0.timestamp() + h * 3600), ws=round(float(WS[i0 + h]), 1), wd=round(float(WD[i0 + h])),
              T=round(float(T[i0 + h]), 1), rh=round(float(RH[i0 + h])), p=round(float(P[i0 + h]), 1), m1=round(float(m1[i0 + h]), 1))
         for h in range(TSTOP_H + 1)]
foci = []
for f in FOCI:
    x0, y0 = lonlat_to_px(f[5], f[8]); x1, y1 = lonlat_to_px(f[6], f[7])
    foci.append(dict(id=f[0], n=f[1], first=f[2], last=f[9], t_first=round(hrs(f[2]), 2), t_last=round(hrs(f[9]), 2),
                     ign=lonlat_to_px(f[3], f[4]), lonlat=[f[3], f[4]], box=[min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]))
area = {f"{a:g}": [round(float(((toa[a] > 0) & (toa[a] <= h)).sum() * CELL ** 2 / 1e6), 1) for h in range(TSTOP_H + 1)] for a in toa}
json.dump(dict(t0=T0.isoformat(), n=N // k, cell_m=CELL * k, corners_lonlat=[list(p) for p in zip(*crn)],
               fuel=[int(v) for v in fbc.ravel()], toa={f"{a:g}": coarse_toa(toa[a]) for a in toa},
               hours=hours, foci=foci, area_km2=area),
          open(os.path.join(HERE, "viz_data.json"), "w"), separators=(",", ":"))
print("готово: viz_data.json", os.path.getsize(os.path.join(HERE, "viz_data.json")) // 1024, "КБ")
