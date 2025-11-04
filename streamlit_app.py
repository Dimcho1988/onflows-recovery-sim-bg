import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from recovery_model import (
    zone_spill_matrix, acute_ratios, effective_acute,
    normalize_subjective, subj_multiplier, obj_multiplier,
    readiness_initial, Dz_eff, ready_curve, summarize,
    DEFAULT_SPILL, ETA0, ALPHA0
)
from data_sim import simulate_zone_loads, simulate_subjective, simulate_objective

st.set_page_config(page_title="onFlows • Възстановяване по зони (демо)", layout="wide")

st.title("onFlows • Зонален модел за възстановяване (0–100%) — демо")

# ====== Sidebar ======
with st.sidebar:
    st.header("Симулация и параметри")
    n_zones = st.number_input("Брой зони", 3, 8, 6, 1)
    base_scale = st.slider("Мащаб на хроничната база", 0.5, 3.0, 1.0, 0.1)

    st.subheader("Разлив между зони (spill)")
    w0 = st.number_input("|Δ|=0 (самата зона)", 0.0, 2.0, DEFAULT_SPILL[0], 0.05)
    w1 = st.number_input("|Δ|=1 (съседни)", 0.0, 1.0, DEFAULT_SPILL[1], 0.05)
    w2 = st.number_input("|Δ|=2", 0.0, 1.0, DEFAULT_SPILL[2], 0.05)
    spill = {0:w0, 1:w1, 2:w2}

    st.subheader("Константи на модела")
    eta0 = st.slider("eta0 (влияе върху R0)", 0.05, 0.5, ETA0, 0.01)
    alpha0 = st.slider("alpha0 (дни до възстановяване)", 0.5, 4.0, ALPHA0, 0.1)

    st.subheader("Днешна болезненост по зони (cpain множител)")
    cpain = [st.slider(f"Z{z+1}", 1.0, 1.5, 1.0, 0.01) for z in range(n_zones)]

    st.divider()
    st.markdown("### Режим вход: Авто + Ръчно")
    if st.button("Генерирай авто данни"):
        st.session_state["today_base"] = simulate_zone_loads(n_zones, base_scale=base_scale)
        st.session_state["subj"] = simulate_subjective()
        st.session_state["obj"] = simulate_objective()

# Инициализация (ако няма още данни)
if "today_base" not in st.session_state:
    st.session_state["today_base"] = simulate_zone_loads(n_zones, base_scale=base_scale)
if "subj" not in st.session_state:
    st.session_state["subj"] = simulate_subjective()
if "obj" not in st.session_state:
    st.session_state["obj"] = simulate_objective()

today, base21 = st.session_state["today_base"]
subj = st.session_state["subj"]
obj = st.session_state["obj"]

# ====== Manual overrides ======
st.subheader("Входни данни")
c1, c2, c3 = st.columns([1.2, 1.2, 1.0])

with c1:
    st.markdown("**Натоварване по зони (произволни единици)**")
    df_loads = pd.DataFrame({"Зона":[f"Z{z+1}" for z in range(n_zones)],
                             "L_today": today[:n_zones].astype(float),
                             "L_21d": base21[:n_zones].astype(float)})
    df_edit = st.data_editor(df_loads, use_container_width=True, num_rows="dynamic")
    today = df_edit["L_today"].to_numpy()
    base21 = df_edit["L_21d"].to_numpy()

with c2:
    st.markdown("**Субективни (0–10)**")
    subj["fatigue"] = st.slider("Умора", 0, 10, int(subj["fatigue"]))
    subj["doms"] = st.slider("DOMS", 0, 10, int(subj["doms"]))
    subj["sleep"] = st.slider("Качество на съня", 0, 10, int(subj["sleep"]))
    subj["stress"] = st.slider("Стрес", 0, 10, int(subj["stress"]))
    subj["desire"] = st.slider("Желание за тренировка", 0, 10, int(subj["desire"]))
    subj["freshness"] = st.slider("Свежест", 0, 10, int(subj["freshness"]))

with c3:
    st.markdown("**Обективни**")
    obj["hrv_z"] = st.number_input("HRV z-score (по-ниско по-лошо)", -3.0, 3.0, float(obj["hrv_z"]), 0.1)
    obj["hr_rest_delta"] = st.number_input("ΔHR покой (% над средното)", -2.0, 5.0, float(obj["hr_rest_delta"]), 0.1)
    obj["fi"] = st.number_input("Fatigue Index (0–1.5)", 0.0, 1.5, float(obj["fi"]), 0.05)

# ====== Изчисления ======
W = zone_spill_matrix(n_zones, spill)
rk = acute_ratios(today[:n_zones], base21[:n_zones])
rzeff = effective_acute(rk, W)

subj_norm = normalize_subjective(subj)
M_subj = subj_multiplier(subj_norm={k:subj_norm[k] for k in subj_norm})
M_obj = obj_multiplier(obj)

R0 = readiness_initial(rzeff, M_obj, M_subj, eta0)
Dz = Dz_eff(rk, W, M_obj, M_subj, np.array(cpain[:n_zones]), alpha0)

# ====== Графики ======
st.subheader("Динамика на готовността")
horizon_days = st.slider("Хоризонт за графиката (дни)", 1.0, 14.0, 7.0, 0.5)
t = np.linspace(0.0, horizon_days, int(horizon_days*24)+1)
Rcurve = ready_curve(t, R0, Dz)

fig = go.Figure()
for z in range(n_zones):
    fig.add_trace(go.Scatter(x=t, y=Rcurve[z], mode="lines", name=f"Z{z+1}"))
fig.update_layout(xaxis_title="Дни след стимула", yaxis_title="Готовност %", yaxis=dict(range=[0,100]))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Числови справки")
target_p = st.slider("Целева готовност (%)", 50, 100, 70, 1)
table = summarize(n_zones, rzeff, R0, Dz, p_target=target_p)
st.dataframe(table, use_container_width=True)

# Stacked bar: принос на съседни зони към r_eff на избрана зона
st.subheader("Принос от съседни зони (stacked bar)")
z_pick = st.selectbox("Целева зона", [f"Z{z+1}" for z in range(n_zones)], index=2)
z_idx = int(z_pick[1:]) - 1
# принос: W[z,k] * r_k
contrib = W[z_idx,:] * rk
fig2 = go.Figure()
running = np.zeros_like(contrib)
for k in range(n_zones):
    fig2.add_trace(go.Bar(x=[z_pick], y=[contrib[k]], name=f"от Z{k+1}"))
fig2.update_layout(barmode="stack", yaxis_title="Принос към r_eff")
st.plotly_chart(fig2, use_container_width=True)

# Експорт на CSV (времеви профил)
ts_df = pd.DataFrame({"t_days": t})
for z in range(n_zones):
    ts_df[f"Z{z+1}_ready%"] = Rcurve[z]
csv_bytes = ts_df.to_csv(index=False).encode("utf-8")
st.download_button("Експорт: readiness по време (CSV)", data=csv_bytes, file_name="readiness_timeseries.csv", mime="text/csv")

# Формули (LaTeX)
st.subheader("Формули")
st.latex(r"r_k = \frac{L_{k,\,today}}{L_{k,\,21d} + \varepsilon}")
st.latex(r"r^{eff}_z = \sum_k W_{z,k} \cdot r_k")
st.latex(r"R_{0,z} = 1 - \eta_0 \cdot r^{eff}_z \cdot H,\quad H=\tfrac{1}{2}(M_{obj}+M_{subj})")
st.latex(r"D^{eff}_z = \sum_k W_{z,k} (\alpha_0 M_{obj} M_{subj} \cdot r_k) \cdot c_{pain,z}")
st.latex(r"Ready_z(t) = 100\,\Big(1 - (1-R_{0,z}) e^{-\ln(20)\, t / D^{eff}_z}\Big)")

st.caption("Съвет: промени входа вляво и наблюдавай как се местят кривите и времената до целевите проценти.")
