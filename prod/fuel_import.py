"""Своя карта топлива (GeoTIFF с кодами FBFM40, любой охват и разрешение, EPSG:4326) -> lc-файл движка на сетке 500 м.

    python fuel_import.py --tif fueltype_labeled_kz_k40.tif --base ext_kz/lc.npz --meta ext_kz/slices_v3_channels.json --out lc_custom.npz
    затем запуск движка с  FIRE_LC=lc_custom.npz

Что пишется (формат как у lc.npz, движок читает его без правок):
- frac[5]  - доли групп в %, ячейка 500 м: grass (GR, GS), crop (NB3=93), shrub_tree (SH, TU, TL), wetland (нет в FBFM40, 0),
             nonburnable (91, 92, 98, 99). Через них считается скорость в физике (sim_physics.fuel_factor) и каналы U-Net;
- fbfm40   - канал U-Net. Модель училась на старой карте с кодами {99, 101, 102, 104, 121}, поэтому новые коды сводятся к ним
             (GR1->101; GR2, GR3->102; GR4-9->104; GS, SH, TU, TL->121; NB->99). Полный модальный код - в fbfm40_full;
- covered  - как в базе.
Ячейки, где своя карта покрывает < 50 % площади, берутся из базового lc.npz без изменений.
EPSG:6933 цилиндрическая: x зависит только от долготы, y - только от широты, поэтому привязка пикселей считается по строкам/столбцам.
"""
import argparse, json
import numpy as np
import rasterio
from pyproj import Transformer

GROUPS = ["grass", "crop", "shrub_tree", "wetland", "nonburnable"]


def code_tables():
    grp = np.full(256, -1, np.int8); old = np.zeros(256, np.int16)
    for c in range(101, 110): grp[c] = 0; old[c] = 101 if c == 101 else (102 if c <= 103 else 104)
    for c in range(121, 125): grp[c] = 0; old[c] = 121
    for c in list(range(141, 150)) + list(range(161, 166)) + list(range(181, 190)): grp[c] = 2; old[c] = 121
    for c in (91, 92, 98, 99): grp[c] = 4; old[c] = 99
    grp[93] = 1; old[93] = 99
    return grp, old


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--meta", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min_cover", type=float, default=0.5); ap.add_argument("--rows", type=int, default=1024)
    a = ap.parse_args(argv)

    g = json.load(open(a.meta, encoding="utf8"))["grid"]; cell = json.load(open(a.meta, encoding="utf8"))["cell_m"]
    X0, Y1, GW, GH = g["X0"], g["Y1"], g["GW"], g["GH"]
    base = dict(np.load(a.base)); grp, old = code_tables()
    codes = np.flatnonzero(grp >= 0); K = len(codes); kidx = np.full(256, K, np.int64); kidx[codes] = np.arange(K)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)

    with rasterio.open(a.tif) as src:
        assert src.crs.to_epsg() == 4326, f"ожидается EPSG:4326, а не {src.crs}"
        T = src.transform; H, W = src.height, src.width
        lon = T.c + (np.arange(W) + 0.5) * T.a
        cc = ((tr.transform(lon, np.zeros(W))[0] - X0) // cell).astype(np.int64)
        lat_all = T.f + (np.arange(H) + 0.5) * T.e
        rr_all = ((Y1 - tr.transform(np.zeros(H), lat_all)[1]) // cell).astype(np.int64)
        okc = (cc >= 0) & (cc < GW)
        c_lo, c_hi = cc[okc].min(), cc[okc].max() + 1; ok_r = (rr_all >= 0) & (rr_all < GH)
        r_lo, r_hi = rr_all[ok_r].min(), rr_all[ok_r].max() + 1
        nr, nc = r_hi - r_lo, c_hi - c_lo
        cnt = np.zeros((nr * nc, K), np.uint32)
        print(f"карта {W}x{H}, окно сетки r {r_lo}:{r_hi} c {c_lo}:{c_hi}, кодов {K}", flush=True)
        for i0 in range(0, H, a.rows):
            i1 = min(H, i0 + a.rows)
            v = src.read(1, window=((i0, i1), (0, W)))
            k = kidx[v]; rr = rr_all[i0:i1]
            m = (k < K) & okc[None, :] & ok_r[i0:i1, None]
            if not m.any(): continue
            cid = (rr[:, None] - r_lo) * nc + (cc[None, :] - c_lo)
            flat = cid[m] * K + k[m]
            u, n = np.unique(flat, return_counts=True)
            cnt.reshape(-1)[u] += n.astype(np.uint32)
            if (i0 // a.rows) % 10 == 0: print(f"  строки {i1}/{H}", flush=True)

    tot = cnt.sum(1).astype(np.float64)
    full = np.full(nr * nc, -1, np.int16); has = tot > 0
    full[has] = codes[cnt[has].argmax(1)]
    gfr = np.zeros((5, nr * nc), np.float64)
    for j in range(5): gfr[j] = cnt[:, grp[codes] == j].sum(1)
    gfr = np.where(has, gfr / np.maximum(tot, 1), 0)

    # доля площади ячейки, покрытая картой: пикселей 20 м в ячейке 500 м ~ (cell / размер пикселя)^2
    px_m = abs(T.a) * 111320 * np.cos(np.deg2rad(np.clip(np.abs(lat_all).mean(), 0, 80)))
    px_area = px_m * abs(T.e) * 110540
    cover = tot * px_area / cell ** 2
    use = (cover >= a.min_cover).reshape(nr, nc)

    frac = base["frac"].copy(); fb = base["fbfm40"].copy(); fbf = base["fbfm40"].copy()
    win = (slice(r_lo, r_hi), slice(c_lo, c_hi))
    newf = np.round(gfr * 100).astype(np.uint8).reshape(5, nr, nc)
    for j in range(5): frac[j][win] = np.where(use, newf[j], frac[j][win])
    fm = full.reshape(nr, nc)
    fb[win] = np.where(use, old[np.clip(fm, 0, 255)], fb[win]); fbf[win] = np.where(use, fm, fbf[win])
    np.savez_compressed(a.out, frac=frac, groups=np.array(GROUPS), fbfm40=fb, fbfm40_full=fbf,
                        covered=base["covered"], custom_mask=np.pad(use, ((r_lo, GH - r_hi), (c_lo, GW - c_hi))))
    print(f"готово: {a.out}; заменено ячеек {int(use.sum())}; доли групп по ним (%):",
          {GROUPS[j]: round(float(newf[j][use].mean()), 1) for j in range(5)})


if __name__ == "__main__":
    main()
