#!/bin/bash
# Итерации 11-12 (19.11): голова площади (шаг 32) и переменный k_ros (шаг 33). Ждёт конца run_iter3.sh (обучение и оценка не параллельно).
cd "$(dirname "$0")"; PY="../../.venv/Scripts/python.exe -X utf8 -W ignore"; mkdir -p logs/iter_done
until grep -qE "all done|FAIL" logs/run_iter3.txt; do sleep 60; done
step() { local id=$1; shift; [ -f logs/iter_done/$id ] && { echo "skip $id"; return 0; }
  echo "== $id $(date +%H:%M)"; env "$@" > logs/iter_$id.txt 2>&1 && touch logs/iter_done/$id && echo "ok $id $(date +%H:%M)" || { echo "FAIL $id"; exit 1; }; }
step it11_all4  UNET=wx,wxbig,wxn,wxbign TAG=all4 $PY scripts/32_area_head.py
grep -q "all done" logs/run_iter3.txt && step it11_all4F UNET=wxF,wxbigF,wxnF,wxbignF TAG=all4F $PY scripts/32_area_head.py
step it12_kdyn  $PY scripts/33_k_dyn.py
echo "all done $(date +%H:%M)"
