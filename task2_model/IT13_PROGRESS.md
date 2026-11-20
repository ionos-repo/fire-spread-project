# Итерация 13+ — журнал для возобновления (после отключения света читать отсюда)
План: C:\Users\Ionos Tech\.claude\plans\imperative-toasting-puzzle.md
Прод сейчас: it12 all4kdyn (out_v3/prod_hybrid.json, k_dyn.joblib). Бэкап до it12: prod_hybrid_before_kdyn.json.

## Шаги
- [x] 1a. k по местности v2 (+почва, VPD, HDW): `KV=2 python 33_k_dyn.py` -> out_v3/k_dyn_v2.{csv,joblib}, k_dyn_c_*_v2.npy (кэш признаков k_dyn_cache_*.npz переиспользуется)
  ИТОГ 1a: физика тест 0.057/0.100/0.126/0.127 = как v1 (0.057/0.101/0.126/0.127). Отклонено, прод не трогаем.
- [x] 1b. Классификатор «взрывного роста» 34_blowup.py
  ИТОГ 1b: AUC большого роста тест 0.73–0.86, заповедники 0.75–0.85 (out_v3/blowup.csv, blow_<part>.npy)
  1c: запуск `bash run_iter6.sh` (возобновляемый, logs/iter_done)
- [x] 1c. Гибрид с новым k / p_blow (26_iter_eval.py), сравнить с it12 на val
  ИТОГ 1c: гибрид+blow val 0.311/0.314/0.331/0.337 vs it12 0.313/0.316/0.330/0.336; тест 0.248/0.246/0.287/0.306 vs 0.250/0.249/0.287/0.308; заповедники 0.289/0.425/0.477/0.503 vs 0.289/0.422/0.466/0.497. По val прироста нет -> ОТКЛОНЕНО, прод без blow.
- [~] 2. Переменный k в активном режиме
  Офлайн (35_k_active_check.py, растущие срезы): тест 0.086/0.132/0.156/0.164 -> 0.106/0.161/0.178/0.183. Код в engine (k_dyn_active ref=0.7138 mult=1.589) и prod_hybrid.json; бэкап prod_hybrid_before_kact.json. ОСТАЛОСЬ: prod/checks.py после eval 1c (не параллельно)
- [~] 3. Онлайн-ассимиляция FIRMS на сайте
  Код: engine.online_c (физика с t−24 ч, c по наблюдённому росту, вес w_online=0.4), live.forecast передаёт c_online, в ответе k_ros/k_online. live.db только 03–10.11.2026 -> ретроспектива слабая; офлайн-оценка it12: +0.01/+0.005 (12/24 ч). ОСТАЛОСЬ: smoke-тест live.forecast на 2 пожарах после eval
- [~] 4. Контуры P10/P50/P90
  Резидуал log c* − log c_модели (растущие val): P10/P50/P90 ×0.19/0.53/1.46. Код: engine band (сдвиг log(T/h)), live pred_p10_km2/pred_p90_km2/pred_p90, сайт: пунктир P90 + «диапазон». Конфиг k_band в prod_hybrid.json. ОСТАЛОСЬ: checks + smoke
- [ ] 5. prod/checks.py (лог prod/checks_log_it13.txt) — запущено
- [x] 6. k монотонный по ветру (проверка 2 «площадь растёт с ветром» упала: штиль 65 км² > 6 м/с 45): MONO=1 33 -> k_dyn_mono.joblib, 36_kdyn_T.py -> phys_T_*_kdyn_mono.npy (готово), физика тест 0.058/0.102/0.125/0.125 (=). ДАЛЕЕ: run_iter6.sh (it13_mono), 35 с KSFX=_mono, конфиг, checks
  ИТОГ 6: гибрид mono val 0.313/0.315/0.330/0.336 (= it12), тест 0.249/0.248/0.286/0.307, заповедники 0.295/0.425/0.465/0.498. ПРИНЯТО. Экспорт: k_dyn_mono.joblib, k_dyn_active ref=0.7323 mult=1.511, k_band 0.18/1.43. Бэкап prod_hybrid_before_it13mono.json. ДАЛЕЕ: checks (prod/checks_log_it13b.txt), smoke live.forecast
