import numpy as np
import pandas as pd

RNG = np.random.default_rng(123)

def simulate_zone_loads(n_zones:int=6, days:int=21, base_scale:float=1.0):
    """
    Връща:
      today: днешно натоварване по зони (единици произволни)
      base21: "хронично" ниво по зони (последни ~21 дни)
    """
    base_raw = RNG.gamma(shape=2.0, scale=base_scale, size=(days, n_zones))
    base21 = base_raw.mean(axis=0) * days * 0.5 + RNG.uniform(0.2, 1.0, size=n_zones)
    today = RNG.gamma(shape=2.0, scale=1.0, size=n_zones)
    focus = RNG.choice(n_zones, size=RNG.integers(1,3), replace=False)
    today *= 0.6
    today[focus] *= float(RNG.uniform(2.0, 4.0))
    return today, base21

def simulate_subjective():
    # 0-10 скали
    return {
        "fatigue":  int(RNG.integers(3, 9)),
        "doms":     int(RNG.integers(2, 8)),
        "sleep":    int(RNG.integers(3, 9)),
        "stress":   int(RNG.integers(2, 8)),
        "desire":   int(RNG.integers(3, 9)),
        "freshness":int(RNG.integers(3, 9)),
    }

def simulate_objective():
    # hrv z ~ N(0,1); hr_rest_delta в %; FI ~ [0..1.5]
    return {
        "hrv_z": float(RNG.normal(0.0, 0.8)),
        "hr_rest_delta": float(np.clip(RNG.normal(0.0, 0.5), -1.0, 2.0)),
        "fi": float(np.clip(RNG.normal(0.3, 0.2), 0.0, 1.5)),
    }
