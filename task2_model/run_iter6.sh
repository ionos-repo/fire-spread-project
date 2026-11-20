#!/bin/bash
# it13: гибрид + признаки «взрывного роста»; it12 база пересчитывается с val для сравнения. Возобновляемый (logs/iter_done).
cd "$(dirname "$0")"; PY="../../.venv/Scripts/python.exe -X utf8 -W ignore"; mkdir -p logs/iter_done
step() { local id=$1; shift; [ -f logs/iter_done/$id ] && { echo "skip $id"; return 0; }
  echo "== $id $(date +%H:%M)"; env "$@" > logs/iter_$id.txt 2>&1 && touch logs/iter_done/$id && echo "ok $id $(date +%H:%M)" || { echo "FAIL $id"; exit 1; }; }
U=wx_tta,wxbig_tta,wxn_tta,wxbign_tta
[ -f out_v3/blow_reserves.npy ] || (cd scripts && KV=2 ../$PY 34_blowup.py > ../out_v3/log_blowup.txt 2>&1)
step it13_base  ITER=12 NAME=all4kdyn UNET=$U HEAD=hybrid PHYS_SFX=_kdyn HYB_VAL=1 $PY scripts/26_iter_eval.py
step it13_blow  ITER=13 NAME=all4kdynblow UNET=$U HEAD=hybrid PHYS_SFX=_kdyn HYB_VAL=1 BLOW=1 $PY scripts/26_iter_eval.py
step it13_mono  ITER=13 NAME=all4kdynmono UNET=$U HEAD=hybrid PHYS_SFX=_kdyn_mono HYB_VAL=1 $PY scripts/26_iter_eval.py
echo "all done $(date +%H:%M)"
