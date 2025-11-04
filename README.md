# onflows-recovery-sim (BG) — демо на зонален модел за възстановяване

Това е тестово Streamlit приложение (браузър), което симулира *поток от данни* и визуализира динамиката на възстановяване (0–100%) по зони Z1–Z6.

## Какво включва
- Автоматична симулация + възможност за ръчна корекция на входа (днешно и 21 дни по зони).
- Субективни (умора, DOMS, сън, стрес, желание, свежест) и обективни (HRV z, ΔHRrest%, FI) показатели.
- Разлив между зони (spill) с настраиваеми тежести.
- Графика на готовността, таблични справки, *stacked bar* на приноса от съседни зони, формули (LaTeX) и експорт в CSV.

## Стартиране локално
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Публикуване в Streamlit Cloud (GitHub → Share)
1) Създай публично репо в GitHub, напр. `onflows-recovery-sim-bg`.
2) Качи всички файлове от тази папка (коренът съдържа `streamlit_app.py`).
3) Влез в https://share.streamlit.io → Create app → избери репото, branch `main`, файл `streamlit_app.py` → Deploy.
4) При всяка промяна в GitHub – Cloud прере-деплойва автоматично.

## Файлове
- `streamlit_app.py` — UI и графики (на български)
- `recovery_model.py` — формули и помощни функции
- `data_sim.py` — генератор на синтетични данни
- `requirements.txt`, `README.md`
