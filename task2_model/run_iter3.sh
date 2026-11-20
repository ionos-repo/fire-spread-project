#!/bin/bash
# Итерация 10 (ITERATIONS.md, 19.11): дообучение 4 U-Net прода на крупных пожарах. Возобновляемо (logs/iter_done/<шаг>).
# Запуск: bash run_iter3.sh > logs/run_iter3.txt 2>&1
cd "$(dirname "$0")"; PY="../../.venv/Scripts/python.exe -X utf8 -W ignore"; mkdir -p logs/iter_done
step() { local id=$1; shift; [ -f logs/iter_done/$id ] && { echo "skip $id"; return 0; }
  echo "== $id $(date +%H:%M)"; env "$@" > logs/iter_$id.txt 2>&1 && touch logs/iter_done/$id && echo "ok $id $(date +%H:%M)" || { echo "FAIL $id"; exit 1; }; }
FT="EXTRA=trainx BIG_OS=10:3,50:6 LR=5e-4 EPOCHS=6 VAL_SRC=gfsa"
step it10_wx     RUN=wxF     INIT=wx     $FT $PY scripts/07_train_v3.py
step it10_wxbig  RUN=wxbigF  INIT=wxbig  $FT $PY scripts/07_train_v3.py
step it10_wxn    RUN=wxnF    INIT=wxn    AUG_GFS=1 $FT $PY scripts/07_train_v3.py
step it10_wxbign RUN=wxbignF INIT=wxbign AUG_GFS=1 $FT $PY scripts/07_train_v3.py
step it10_pred   RUNS=wxF,wxbigF,wxnF,wxbignF $PY scripts/25_iter_predict.py
step it10_diag   UNET=wxF,wxbigF,wxnF,wxbignF TAG=all4F $PY scripts/31_size_diag.py
step it10_eval   ITER=10 NAME=all4F UNET=wxF_tta,wxbigF_tta,wxnF_tta,wxbignF_tta HEAD=unet,hybrid $PY scripts/26_iter_eval.py
echo "all done $(date +%H:%M)"
