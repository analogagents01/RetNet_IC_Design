import sys

import numpy as np
import pandas as pd
from scipy import stats


def per_split_mape(path):
    raw = pd.read_csv(path)
    name = raw["model"].iloc[0] if "model" in raw else path
    return name, raw.groupby("repeat")["MAPE"].mean().sort_index()


def pitman_morgan(a, b):
    r, p = stats.pearsonr(a + b, a - b)
    return r, p


def sign_flip(a, b):
    stat = lambda x, y: abs(np.log(x.std(ddof=1) / y.std(ddof=1)))
    obs, J = stat(a, b), len(a)
    masks = ((np.arange(2 ** J)[:, None] >> np.arange(J)) & 1).astype(bool)
    null = np.array([stat(np.where(m, b, a), np.where(m, a, b)) for m in masks])
    return float(np.mean(null >= obs - 1e-12))


def boot_ci(a, b, n=10000, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(a), (n, len(a)))
    sa, sb = a[idx].std(axis=1, ddof=1), b[idx].std(axis=1, ddof=1)
    ok = sb > 0
    return np.percentile(sa[ok] / sb[ok], [2.5, 97.5])


def holm(p):
    p = np.asarray(p, float)
    adj, running = np.empty_like(p), 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adj[idx] = running
    return adj


def main(paths):
    if len(paths) < 2 or len(paths) % 2:
        raise SystemExit(__doc__)
    results = []
    for pa, pb in zip(paths[::2], paths[1::2]):
        na, sa = per_split_mape(pa)
        nb, sb = per_split_mape(pb)
        common = sa.index.intersection(sb.index)
        a, b = sa[common].values, sb[common].values
        r, p_pm = pitman_morgan(a, b)
        lo, hi = boot_ci(a, b)
        res = dict(A=na, B=nb, J=len(common),
                   SD_A=a.std(ddof=1), SD_B=b.std(ddof=1), SD_ratio=a.std(ddof=1) / b.std(ddof=1),
                   boot_CI_low=lo, boot_CI_high=hi,
                   pitman_morgan_r=r, p_pitman_morgan=p_pm,
                   p_sign_flip=sign_flip(a, b),
                   p_brown_forsythe=stats.levene(a, b, center="median").pvalue)
        results.append(res)
    df = pd.DataFrame(results)
    df["p_pitman_morgan_holm"] = holm(df.p_pitman_morgan)
    pd.set_option("display.width", 200)
    print(df.round(4).T.to_string(header=False))
    df.to_csv("variance_test_results.csv", index=False)
    


if __name__ == "__main__":
    main(sys.argv[1:])
