"""Random forest benchmark on the two-stage Miller op-amp dataset.

One random forest per target (11 models) with a shared configuration. The
data handling, splits, tuning budget and metrics are shared with the other
nine benchmark scripts; only the model section at the end differs.

Usage:
    python rf.py

Results are written to fair_results/. Finished repeats are skipped, so an
interrupted run can simply be restarted.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Shared settings (keep identical across all benchmark scripts)
INPUT_CSV = "design_points_results_combined.csv"
OUT_DIR = "fair_results"
N_TARGETS = 19          # first 19 columns: design parameters
N_INPUTS = 28           # next 28 columns: performance / operating point
N_REPEATS = 10          # splits, seeds 42..51
N_TRIALS = 15           # random-search trials per model and split
TEST_SIZE = 0.20
VAL_FROM_TRAIN = 0.111  # 640 train / 80 val / 180 test
BASE_SEED = 42
IQR_CLIP_K = 4.0

NN_FINAL_EPOCHS, NN_FINAL_PATIENCE = 350, 40
NN_TRIAL_EPOCHS, NN_TRIAL_PATIENCE = 80, 15

GEOM = ["L1", "L3", "L5", "W1", "W3", "W5", "W6", "W7", "IBIAS"]
TARGETS = GEOM + ["Rz", "CC"]
LOG_R2_TARGETS = {"Rz", "CC"}   # R2 on the log10 scale for these two
BLEND_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)


# Data
def load_data():
    df = pd.read_csv(INPUT_CSV)
    if df.shape[1] < N_TARGETS + N_INPUTS:
        raise ValueError(f"{INPUT_CSV}: expected at least {N_TARGETS + N_INPUTS} columns")
    df = df.iloc[:, :N_TARGETS + N_INPUTS]
    tcols, icols = list(df.columns[:N_TARGETS]), list(df.columns[N_TARGETS:])
    df = df.dropna(subset=tcols + icols).reset_index(drop=True)

    X = df[icols].astype(float).copy()
    X.columns = [str(c) for c in X.columns]
    # derived features, identical for every model (34 inputs in total)
    X["PM_over_UGBW"] = X["PM"] / X["UGBW"].replace(0, np.nan)
    X["GBW_div_UGBW"] = X["GBW"] / (X["UGBW"] + 1e-8)
    X["GBW_log"] = np.log10(np.abs(X["GBW"]) + 1e-8)
    X["UGBW_log"] = np.log10(np.abs(X["UGBW"]) + 1e-8)
    X["GBW_squared"] = X["GBW"] ** 2
    X["UGBW_squared"] = X["UGBW"] ** 2
    X = X.fillna(0.0)

    Y = np.log10(df[TARGETS].astype(float).values)
    return X, Y, tcols, icols


def make_split(n, repeat):
    seed = BASE_SEED + repeat
    tr_full, te = train_test_split(np.arange(n), test_size=TEST_SIZE, random_state=seed)
    tr, va = train_test_split(tr_full, test_size=VAL_FROM_TRAIN, random_state=seed)
    return tr, va, te


def preprocess(X, Y, tr):
    # clipping bounds and scalers are fitted on the training rows only
    q1, q3 = X.iloc[tr].quantile(0.25), X.iloc[tr].quantile(0.75)
    iqr = q3 - q1
    Xc = X.clip(lower=q1 - IQR_CLIP_K * iqr, upper=q3 + IQR_CLIP_K * iqr, axis=1)
    sx = StandardScaler().fit(Xc.iloc[tr].values)
    sy = StandardScaler().fit(Y[tr])
    return sx.transform(Xc.values), sy.transform(Y), sy


# Metrics
def target_r2(t, ylog_true, ylog_pred):
    if t in LOG_R2_TARGETS:
        return r2_score(ylog_true, ylog_pred)
    return r2_score(10.0 ** ylog_true, 10.0 ** ylog_pred)


def mean_r2(ylog_true, ylog_pred):
    return float(np.mean([target_r2(t, ylog_true[:, j], ylog_pred[:, j])
                          for j, t in enumerate(TARGETS)]))


def all_metrics(ylog_true, ylog_pred):
    # errors in physical units (m, A, ohm, F)
    rows = []
    for j, t in enumerate(TARGETS):
        a, p = 10.0 ** ylog_true[:, j], 10.0 ** ylog_pred[:, j]
        rows.append({"target": t,
                     "R2": target_r2(t, ylog_true[:, j], ylog_pred[:, j]),
                     "RMSE": float(np.sqrt(mean_squared_error(a, p))),
                     "MAE": float(mean_absolute_error(a, p)),
                     "MAPE": float(np.mean(np.abs((a - p) / a)) * 100),
                     "MedAE": float(median_absolute_error(a, p)),
                     "MaxErr": float(np.max(np.abs(a - p)))})
    return rows


def blend_weights(yva_true_log, pa_log, pb_log):
    # per-target blend weight, selected on the validation rows
    ws = []
    for j, t in enumerate(TARGETS):
        scores = [target_r2(t, yva_true_log[:, j], w * pa_log[:, j] + (1 - w) * pb_log[:, j])
                  for w in BLEND_GRID]
        ws.append(float(BLEND_GRID[int(np.argmax(scores))]))
    return np.array(ws)


# Tuning and evaluation loop
def tune(model, Xtr, Ytr, Xva, Yva, sy, seed, log):
    # trial 0 is the default config; score is the mean validation R2
    if N_TRIALS <= 0:
        return model.default_config(), None
    rng = np.random.default_rng(seed)
    yva_log = sy.inverse_transform(Yva)
    best_cfg, best = None, -np.inf
    for k in range(N_TRIALS):
        cfg = model.default_config() if k == 0 else model.sample_config(rng)
        t0 = time.time()
        pv, _, _ = model.fit_predict(cfg, Xtr, Ytr, Xva, Yva, Xva[:1], seed, sy=sy, trial=True)
        s = mean_r2(yva_log, sy.inverse_transform(pv))
        log(f"  trial {k + 1:2d}/{N_TRIALS}  val R2={s:.4f}  ({time.time() - t0:.0f}s)  {cfg}")
        if s > best:
            best_cfg, best = cfg, s
    return best_cfg, best


def run_model(model):
    os.makedirs(OUT_DIR, exist_ok=True)
    name = model.name
    raw_path = os.path.join(OUT_DIR, f"{name}_raw_results.csv")
    info_path = os.path.join(OUT_DIR, f"{name}_run_info.csv")
    pred_dir = os.path.join(OUT_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    log_path = os.path.join(OUT_DIR, f"{name}_log.txt")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    done = set(pd.read_csv(info_path)["repeat"]) if os.path.exists(info_path) else set()
    X, Y, tcols, icols = load_data()
    log(f"Model: {name}")
    log(f"{len(X)} rows, {len(icols)} raw + 6 derived inputs, targets: {TARGETS}")
    log(f"{N_REPEATS} repeats (640/80/180), {N_TRIALS} tuning trials per repeat")
    t_start = time.time()

    for r in range(N_REPEATS):
        if r in done:
            log(f"[repeat {r}] done, skipping")
            continue
        tr, va, te = make_split(len(X), r)
        Xs, Ys, sy = preprocess(X, Y, tr)
        seed = BASE_SEED + r
        log(f"\n[repeat {r}] seed {seed}")

        if getattr(model, "from_saved", False):
            pv, pt_all, size, cfg, vbest, t_tune, t_fit = model.from_saved(r, Y, va, pred_dir)
            if pv is None:
                log(f"[repeat {r}] {model.missing_msg}")
                continue
        else:
            t0 = time.time()
            cfg, vbest = tune(model, Xs[tr], Ys[tr], Xs[va], Ys[va], sy, seed, log)
            t_tune = time.time() - t0
            t1 = time.time()
            # train and test rows are predicted together (train gives the train-test gap)
            pv, pt_all, size = model.fit_predict(cfg, Xs[tr], Ys[tr], Xs[va], Ys[va],
                                                 np.vstack([Xs[tr], Xs[te]]), seed, sy=sy, trial=False)
            t_fit = time.time() - t1
            pv, pt_all = sy.inverse_transform(pv), sy.inverse_transform(pt_all)
            if hasattr(model, "extra_info"):
                cfg = {**cfg, **model.extra_info()}

        ptr, pte = pt_all[:len(tr)], pt_all[len(tr):]
        np.savez(os.path.join(pred_dir, f"{name}_r{r}.npz"), val=pv, test=pte, train=ptr)

        df_r = pd.DataFrame(all_metrics(Y[te], pte))
        df_r.insert(0, "model", name)
        df_r.insert(0, "repeat", r)
        df_r.to_csv(raw_path, mode="a", header=not os.path.exists(raw_path), index=False)

        train_r2 = mean_r2(Y[tr], ptr)
        pd.DataFrame([{"repeat": r, "model": name, "config": json.dumps(cfg),
                       "val_R2_best_trial": vbest, "train_mean_R2": train_r2,
                       "test_mean_R2": df_r.R2.mean(), "size": size,
                       "tune_seconds": round(t_tune, 1), "fit_seconds": round(t_fit, 1)}]
                     ).to_csv(info_path, mode="a", header=not os.path.exists(info_path), index=False)
        log(f"[repeat {r}] test R2={df_r.R2.mean():.4f}  MAPE={df_r.MAPE.mean():.2f}%  "
            f"train R2={train_r2:.4f}  (tune {t_tune:.0f}s, fit {t_fit:.0f}s)")

    report(name, raw_path, info_path, log)
    log(f"\nTotal time {time.time() - t_start:.1f}s")


def report(name, raw_path, info_path, log):
    if not os.path.exists(raw_path):
        return
    raw, info = pd.read_csv(raw_path), pd.read_csv(info_path)
    n = raw["repeat"].nunique()
    log(f"\nTest results for {name}, mean ± std over {n} repeats")
    log(f"{'target':16s} {'R2':>17s} {'RMSE':>12s} {'MAE':>12s} {'MAPE (%)':>15s} {'MedAE':>12s} {'MaxErr':>12s}")
    rows = []
    for t in TARGETS:
        d = raw[raw.target == t]
        lbl = f"{t} (log10 R2)" if t in LOG_R2_TARGETS else t
        log(f"{lbl:16s} {d.R2.mean():8.4f}±{d.R2.std():.4f} {d.RMSE.mean():12.4g} {d.MAE.mean():12.4g} "
            f"{d.MAPE.mean():7.2f}±{d.MAPE.std():5.2f} {d.MedAE.mean():12.4g} {d.MaxErr.mean():12.4g}")
        rows.append({"target": t, "R2_mean": d.R2.mean(), "R2_std": d.R2.std(),
                     "RMSE": d.RMSE.mean(), "MAE": d.MAE.mean(),
                     "MAPE_mean": d.MAPE.mean(), "MAPE_std": d.MAPE.std(),
                     "MedAE": d.MedAE.mean(), "MaxErr": d.MaxErr.mean()})

    per_rep = raw.groupby("repeat").agg(R2=("R2", "mean"), MAPE=("MAPE", "mean"))
    log(f"\nmean over targets: R2 {per_rep.R2.mean():.4f} ± {per_rep.R2.std():.4f}, "
        f"MAPE {per_rep.MAPE.mean():.2f} ± {per_rep.MAPE.std():.2f}%")
    gap = info.train_mean_R2 - info.test_mean_R2
    log(f"train R2 {info.train_mean_R2.mean():.4f}, test R2 {info.test_mean_R2.mean():.4f}, "
        f"gap {gap.mean():.4f}")
    log(f"median size: {info['size'].median():,.0f} {getattr(sys.modules['__main__'], 'SIZE_UNIT', '')}")
    log(f"time per repeat: tune {info.tune_seconds.mean():.0f}s, fit {info.fit_seconds.mean():.0f}s")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"{name}_test_metrics.csv"), index=False)


# Tree-model helpers (shared by the tree scripts)
def _per_target_fit(make_est, Xtr, Ytr, Xva, Xte):
    # one estimator per target, same config for all targets
    pv, pt, size = [], [], 0
    for j in range(Ytr.shape[1]):
        est = make_est()
        est.fit(Xtr, Ytr[:, j])
        pv.append(est.predict(Xva))
        pt.append(est.predict(Xte))
        size += _tree_size(est)
    return np.column_stack(pv), np.column_stack(pt), size


def _tree_size(est):
    # model size = total number of leaves over all trees
    try:  # random forest
        return int(sum(e.tree_.n_leaves for e in est.estimators_))
    except Exception:
        pass
    try:  # XGBoost
        d = est.get_booster().trees_to_dataframe()
        return int((d["Feature"] == "Leaf").sum())
    except Exception:
        pass
    try:  # LightGBM
        return int(sum(t["num_leaves"] for t in est.booster_.dump_model()["tree_info"]))
    except Exception:
        pass
    try:  # histogram gradient boosting
        return int(sum(p.get_n_leaf_nodes() for it in est._predictors for p in it))
    except Exception:
        return 0


# Model: random forest
from sklearn.ensemble import RandomForestRegressor

SIZE_UNIT = "leaves (11 models, one per target)"


class RFModel:
    name = "rf"

    def default_config(self):
        return {"n_estimators": 500, "max_depth": None, "max_features": "sqrt",
                "min_samples_split": 2, "min_samples_leaf": 1}

    def sample_config(self, rng):
        # lists with None or mixed types are indexed rather than passed to rng.choice
        return {"n_estimators": int(rng.choice([200, 300, 500])),
                "max_depth": [None, 10, 20, 30][rng.integers(4)],
                "max_features": ["sqrt", 0.5, 0.7, 1.0][rng.integers(4)],
                "min_samples_split": int(rng.choice([2, 5])),
                "min_samples_leaf": int(rng.choice([1, 2, 4]))}

    def fit_predict(self, cfg, Xtr, Ytr, Xva, Yva, Xte, seed, **_):
        def make():
            return RandomForestRegressor(**cfg, random_state=seed, n_jobs=-1)
        return _per_target_fit(make, Xtr, Ytr, Xva, Xte)


if __name__ == "__main__":
    run_model(RFModel())
