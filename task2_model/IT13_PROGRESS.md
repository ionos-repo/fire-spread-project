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
- [ ] 2. Переменный k в активном режиме
- [ ] 3. Онлайн-ассимиляция FIRMS на сайте
- [ ] 4. Контуры P10/P50/P90
