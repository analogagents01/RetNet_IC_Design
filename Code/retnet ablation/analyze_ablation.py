import argparse
import glob
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

OUT = "ablation_results"
ORDER = ["full", "no_blocks", "no_retention", "bidirectional", "no_decay", "shuffled"]


def corrected_t(a, b, n_train=640, n_test=180):
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / math.sqrt((1 / J + n_test / n_train) * var)
    return t, 2 * stats.t.sf(abs(t), J - 1)


def holm(p):
    p = np.asarray(p, float)
    adj, running = np.empty_like(p), 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adj[idx] = running
    return adj


def per_split(raw):
    return raw.groupby("repeat")["MAPE"].mean().sort_index()


def main(form):
    files = sorted(glob.glob(os.path.join(OUT, f"{form}_*_raw.csv")))
    if not files:
        raise SystemExit(f"no results in {OUT}/ for form '{form}' - run run_ablation.py first")
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw["base"] = raw["variant"].str.split(":").str[0]

    #reproduction check against your original run
    orig_path = next((p for p in (os.path.join("fair_results_origmask", f"retnet_{form}_raw_results.csv"),
                                  f"retnet_{form}_raw_results.csv") if os.path.exists(p)), None)
    full = per_split(raw[raw.variant == "full"])
    if orig_path and len(full):
        orig = per_split(pd.read_csv(orig_path))
        rep = pd.DataFrame({"original": orig, "rerun_full": full}).dropna()
        rep["diff"] = rep.rerun_full - rep.original
        rep.to_csv(os.path.join(OUT, f"{form}_reproduction.csv"))
        print("Reproduction check (per-split mean MAPE, %):")
        print(rep.round(3).to_string())
        print(f"original {rep.original.mean():.2f} +- {rep.original.std():.2f}   "
              f"re-run {rep.rerun_full.mean():.2f} +- {rep.rerun_full.std():.2f}   "
              f"max |diff| {rep['diff'].abs().max():.2f} points\n")

    #per-split MAPE per variant (shuffled orders averaged per split)
    rows, splits = [], {}
    for base in ORDER:
        sub = raw[raw.base == base]
        if sub.empty:
            continue
        if base == "shuffled":
            per_order = sub.groupby(["variant", "repeat"])["MAPE"].mean().unstack(0)
            s = per_order.mean(axis=1)
            order_means = per_order.mean(axis=0)
            extra = f"orders: {order_means.min():.2f}-{order_means.max():.2f}%"
        else:
            s, extra = per_split(sub), ""
        splits[base] = s
        rows.append({"variant": base, "n_splits": len(s),
                     "params_median": int(sub.groupby("repeat")["n_params"].first().median()),
                     "MAPE_mean": s.mean(), "MAPE_std": s.std(), "note": extra})
    tab = pd.DataFrame(rows).set_index("variant")

    if "full" in splits:
        comps = [v for v in tab.index if v != "full"]
        pr = []
        for v in comps:
            common = splits[v].index.intersection(splits["full"].index)
            a, b = splits[v][common], splits["full"][common]
            t, p = corrected_t(a, b)
            tab.loc[v, "delta_vs_full"] = a.mean() - b.mean()
            tab.loc[v, "better_than_full"] = f"{int((a < b).sum())}/{len(common)}"
            tab.loc[v, "p_raw"] = p
            pr.append(p)
        if pr:
            tab.loc[comps, "p_holm"] = holm(pr)
    tab.to_csv(os.path.join(OUT, f"{form}_ablation_table.csv"))
    print("Ablation (mean MAPE over splits, %):")
    print(tab.round(4).to_string(), "\n")

    pt = raw.groupby(["base", "target"])["MAPE"].mean().unstack(1)
    targets = ["L1", "L3", "L5", "W1", "W3", "W5", "W6", "W7", "IBIAS", "Rz", "CC"]
    pt = pt[[t for t in targets if t in pt.columns]].reindex([b for b in ORDER if b in pt.index])
    pt.to_csv(os.path.join(OUT, f"{form}_ablation_pertarget.csv"))
    print("Per-target mean MAPE (%):")
    print(pt.round(2).to_string(), "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", choices=["joint", "sep"], default="joint")
    main(ap.parse_args().form)
