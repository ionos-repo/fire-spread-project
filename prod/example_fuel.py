"""Прогноз на 12-72 ч от точечного очага при заданной погоде (без интернета) - сравнить базовое топливо и своё.
    python -X utf8 -m prod.example_fuel 80.3 50.3            # базовая lc.npz
    FIRE_LC=lc_custom.npz python -X utf8 -m prod.example_fuel 80.3 50.3
Аргументы: lon lat [ветер м/с] [откуда дует, град]."""
import os, sys
import pandas as pd
from .checks import scen
from .engine import FireEngine
from .config import HZ

lon, lat = float(sys.argv[1]), float(sys.argv[2])
ws = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0; wd = float(sys.argv[4]) if len(sys.argv) > 4 else 270.0
E = FireEngine(); r, c = E.land.rc(lon, lat)
print("топливо:", os.environ.get("FIRE_LC") or "lc.npz (база)")
print("доли групп в ячейке, %:", dict(zip(["трава", "пашня", "кусты/лес", "болото", "негорючее"], E.land.FR[:, r, c].tolist())))
out, _, _ = scen(E, lon, lat, pd.Timestamp("2024-08-10 06:00"), ws, wd, mode="expected")
for k, h in enumerate(HZ):
    print(f"{h:3d} ч: площадь {out['area_km2'][k]:.1f} км2")
