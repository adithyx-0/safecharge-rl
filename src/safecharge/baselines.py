"""Rule baselines FCFS / EDF / LLF. Greedy fill in priority order up to P_grid; return u in [0,1]^N for the env.

They see only what the policy sees: reported departure (never the true one). FCFS uses arrival order only.
"""
import numpy as np


def rule_action(env, kind):
    w = env.view()
    h, occ = w["h"], w["occ"]
    if kind == "FCFS":
        key = w["arr"].astype(float)
    elif kind == "EDF":
        key = w["rep"]
    elif kind == "LLF":
        key = w["laxity"]
    else:
        raise ValueError(f"unknown rule {kind!r}")
    order = [k for k in np.argsort(key, kind="stable") if occ[k]]
    p = np.zeros(env.N)
    left = env.P_grid
    for k in order:
        p[k] = min(h[k], left)
        left -= p[k]
        if left <= 0:
            break
    return np.divide(p, h, out=np.zeros(env.N), where=h > 0)
