import numpy as np
import pandas as pd

# Настройки по подразбиране (променят се от UI)
DEFAULT_SPILL = {0:1.0, 1:0.35, 2:0.15}  # |dz|>=3 -> 0
SUBJECTIVE_WEIGHTS = {
    "fatigue":0.20, "doms":0.20, "sleep":0.15, "stress":0.20, "desire":0.10, "freshness":0.15
}
OBJECTIVE_WEIGHTS = {"hrv_z":0.5, "hr_rest_delta":0.3, "fi":0.2}
KAPPA_SUBJ = 0.15
KAPPA_OBJ = 0.10
CLIP_MIN, CLIP_MAX = 0.85, 1.15
ETA0 = 0.20            # сила на наказанието в R0
ALPHA0 = 1.75          # базов коеф. за дни до възстановяване
LN20 = np.log(20.0)    # ~95% при t=D

def zone_spill_matrix(n_zones:int, spill:dict=None) -> np.ndarray:
    if spill is None:
        spill = DEFAULT_SPILL
    W = np.zeros((n_zones, n_zones))
    for z in range(n_zones):
        for k in range(n_zones):
            dz = abs(z-k)
            W[z,k] = spill.get(dz, 0.0)
    return W

def acute_ratios(today:np.ndarray, base21:np.ndarray, eps:float=1e-6) -> np.ndarray:
    # r_k = L_today / (L_21d + eps)
    return today / (base21 + eps)

def effective_acute(rk:np.ndarray, W:np.ndarray) -> np.ndarray:
    # r_eff[z] = sum_k W[z,k] * r_k
    return (W @ rk)

# ----- Субективни / Обективни -----
def normalize_subjective(subj:dict) -> dict:
    # Карта 0-10 -> [-1,1]; за "позитивните" скали обръщаме знака
    inverted = {"sleep", "desire", "freshness"}
    out = {}
    for k, v in subj.items():
        v = float(v)
        s = (v/10.0)*2 - 1.0  # 0->-1 ; 10->+1
        if k in inverted:
            s = -s
        out[k] = s
    return out

def subj_multiplier(subj_norm:dict, weights:dict=SUBJECTIVE_WEIGHTS, kappa:float=KAPPA_SUBJ) -> float:
    keys = [k for k in weights.keys() if k in subj_norm]
    ssum = sum(weights[k]*subj_norm[k] for k in keys) / (sum(weights[k] for k in keys) + 1e-9)
    M = 1.0 + kappa * ssum
    return float(np.clip(M, CLIP_MIN, CLIP_MAX))

def obj_multiplier(obj:dict, weights:dict=OBJECTIVE_WEIGHTS, kappa:float=KAPPA_OBJ) -> float:
    # HRV z: по-ниско е по-лошо -> взимаме само отрицателната част
    hrv_z = float(obj.get("hrv_z", 0.0))
    hr_delta = max(0.0, float(obj.get("hr_rest_delta", 0.0)))  # %
    fi = max(0.0, float(obj.get("fi", 0.0)))
    bad = weights["hrv_z"] * max(0.0, -hrv_z) + weights["hr_rest_delta"] * hr_delta + weights["fi"] * fi
    M = 1.0 + kappa * bad
    return float(np.clip(M, CLIP_MIN, CLIP_MAX))

def readiness_initial(rzeff:np.ndarray, M_obj:float, M_subj:float, eta0:float=ETA0) -> np.ndarray:
    H = 0.5*(M_obj + M_subj)  # смесване на стреса
    R0 = 1.0 - eta0 * rzeff * H
    return np.clip(R0, 0.0, 1.05)

def alpha_eff(M_obj:float, M_subj:float, alpha0:float=ALPHA0) -> float:
    return alpha0 * M_obj * M_subj

def Dz_eff(rk:np.ndarray, W:np.ndarray, M_obj:float, M_subj:float, cpain:np.ndarray=None,
           alpha0:float=ALPHA0) -> np.ndarray:
    # D_k = alpha_eff * r_k ; разлив към целеви зони и умножение по cpain
    Dk = alpha_eff(M_obj, M_subj, alpha0) * rk
    Dz = (W @ Dk)
    if cpain is not None:
        Dz = Dz * cpain
    return Dz

def ready_curve(t_days:np.ndarray, R0z:np.ndarray, Dz:np.ndarray) -> np.ndarray:
    # Ready%_z(t) = 100 * (1 - (1-R0) * exp(-ln(20)*t/D))
    R = 100.0 * (1.0 - (1.0 - R0z[:,None]) * np.exp(-LN20 * (t_days[None,:] / (Dz[:,None] + 1e-9))))
    return R

def t_for_percent(p:float, R0:float, D:float) -> float:
    p = np.clip(p, 0.0, 100.0) / 100.0
    if p <= R0:
        return 0.0
    ratio = (1.0 - p) / (1.0 - R0 + 1e-9)
    ratio = max(ratio, 1e-9)
    t = - (D / LN20) * np.log(ratio)
    return max(0.0, float(t))

def summarize(n_zones:int, rzeff:np.ndarray, R0:np.ndarray, Dz:np.ndarray, p_target:float=70.0) -> pd.DataFrame:
    rows = []
    for z in range(n_zones):
        tP = t_for_percent(p_target, R0[z], Dz[z])
        rows.append({
            "Зона": f"Z{z+1}",
            "r_eff": round(float(rzeff[z]), 3),
            "R0(%)": round(float(R0[z])*100, 1),
            "D_eff(дни)": round(float(Dz[z]), 2),
            f"t→{int(p_target)}%(ч)": round(tP*24.0, 1),
        })
    return pd.DataFrame(rows)
