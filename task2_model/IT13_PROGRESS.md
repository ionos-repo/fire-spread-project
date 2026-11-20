# Итерация 13+ — журнал для возобновления (после отключения света читать отсюда)
План: C:\Users\Ionos Tech\.claude\plans\imperative-toasting-puzzle.md
Прод сейчас: it12 all4kdyn (out_v3/prod_hybrid.json, k_dyn.joblib). Бэкап до it12: prod_hybrid_before_kdyn.json.

## Шаги
- [x] 1a. k по местности v2 (+почва, VPD, HDW): `KV=2 python 33_k_dyn.py` -> out_v3/k_dyn_v2.{csv,joblib}, k_dyn_c_*_v2.npy (кэш признаков k_dyn_cache_*.npz переиспользуется)
  ИТОГ 1a: физика тест 0.057/0.100/0.126/0.127 = как v1 (0.057/0.101/0.126/0.127). Отклонено, прод не трогаем.
- [x] 1b. Классификатор «взрывного роста» 34_blowup.py
  ИТОГ 1b: AUC большого роста тест 0.73–0.86, заповедники 0.75–0.85 (out_v3/blowup.csv, blow_<part>.npy)
  1c: запуск `bash run_iter6.sh` (возобновляемый, logs/iter_done)
- [ ] 1c. Гибрид с новым k / p_blow (26_iter_eval.py), сравнить с it12 на val
- [~] 2. Переменный k в активном режиме
  Офлайн (35_k_active_check.py, растущие срезы): тест 0.086/0.132/0.156/0.164 -> 0.106/0.161/0.178/0.183. Код в engine (k_dyn_active ref=0.7138 mult=1.589) и prod_hybrid.json; бэкап prod_hybrid_before_kact.json. ОСТАЛОСЬ: prod/checks.py после eval 1c (не параллельно)
- [~] 3. Онлайн-ассимиляция FIRMS на сайте
  Код: engine.online_c (физика с t−24 ч, c по наблюдённому росту, вес w_online=0.4), live.forecast передаёт c_online, в ответе k_ros/k_online. live.db только 03–10.11.2026 -> ретроспектива слабая; офлайн-оценка it12: +0.01/+0.005 (12/24 ч). ОСТАЛОСЬ: smoke-тест live.forecast на 2 пожарах после eval
- [ ] 4. Контуры P10/P50/P90
