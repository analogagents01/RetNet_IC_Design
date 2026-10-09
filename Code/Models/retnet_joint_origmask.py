"""Retention-mask: joint RetNet with the original RetNet decay.

Usage:
    python retnet_joint_origmask.py

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
OUT_DIR = "fair_results_origmask"   
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


# Neural-network training 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_torch(net, cfg, Xtr, Ytr, Xva, Yva, seed, epochs, patience):
    # Huber loss, AdamW, cosine LR, grad clipping, early stopping on val loss
    net = net.to(DEVICE)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Yt = torch.tensor(Ytr, dtype=torch.float32)
    Xv = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    Yv = torch.tensor(Yva, dtype=torch.float32, device=DEVICE)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(Xt, Yt), batch_size=cfg["batch_size"], shuffle=True, generator=g)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = nn.HuberLoss(delta=1.0)

    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()

        net.eval()
        with torch.no_grad():
            vl = crit(net(Xv), Yv).item()
        if vl < best:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    net.load_state_dict(best_state)
    net.eval()
    return net


def torch_predict(net, X):
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()


class NNModel:
    # joint: one network for all 11 targets
    
    joint = True

    def build(self, n_features, n_outputs, cfg):
        raise NotImplementedError

    def fit_predict(self, cfg, Xtr, Ytr, Xva, Yva, Xte, seed, trial=False, **_):
        epochs, patience = ((NN_TRIAL_EPOCHS, NN_TRIAL_PATIENCE) if trial
                            else (NN_FINAL_EPOCHS, NN_FINAL_PATIENCE))
        groups = [list(range(11))] if self.joint else [list(range(9)), [9], [10]]
        pv, pt, n_params = np.zeros((len(Xva), 11)), np.zeros((len(Xte), 11)), 0
        for g in groups:
            torch.manual_seed(seed)
            net = self.build(Xtr.shape[1], len(g), cfg)
            net = train_torch(net, cfg, Xtr, Ytr[:, g], Xva, Yva[:, g], seed, epochs, patience)
            pv[:, g], pt[:, g] = torch_predict(net, Xva), torch_predict(net, Xte)
            n_params += sum(p.numel() for p in net.parameters())
        return pv, pt, n_params


# RetNet
class SimpleRetention(nn.Module):
    def __init__(self, d_model, gamma, head_dim):
        super().__init__()
        self.gamma, self.head_dim = gamma, head_dim
        self.W_Q = nn.Linear(d_model, head_dim, bias=False)
        self.W_K = nn.Linear(d_model, head_dim, bias=False)
        self.W_V = nn.Linear(d_model, head_dim, bias=False)

    def forward(self, X):
        B, N, _ = X.shape
        Q, K, V = self.W_Q(X), self.W_K(X), self.W_V(X)
        idx = torch.arange(N, device=X.device, dtype=X.dtype)
        # original RetNet mask: D[i, j] = gamma^(i - j) for i >= j, else 0
        D = torch.tril(self.gamma ** (idx.unsqueeze(1) - idx.unsqueeze(0)))
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)
        return torch.bmm(scores * D.unsqueeze(0), V)


class MultiScaleRetention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        hd = d_model // n_heads
        gammas = [1.0 - 2.0 ** (-5 - h) for h in range(n_heads)]   # Eq. 9
        self.retentions = nn.ModuleList([SimpleRetention(d_model, g, hd) for g in gammas])
        self.W_G = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.group_norm = nn.GroupNorm(n_heads, d_model)

    def forward(self, X):
        B, N, _ = X.shape
        Y = torch.cat([r(X) for r in self.retentions], dim=-1)
        Y = self.group_norm(Y.reshape(B * N, self.d_model)).reshape(B, N, self.d_model)
        return self.W_O(Y * F.silu(self.W_G(X)))


class RetNetBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout):
        super().__init__()
        self.msr = MultiScaleRetention(d_model, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, X):
        X = X + self.drop(self.msr(self.norm1(X)))
        return X + self.drop(self.ffn(self.norm2(X)))


class RetNetTabular(nn.Module):
    # each input is one token; retention blocks; flatten -> 256 -> outputs
    def __init__(self, n_features, n_outputs, cfg):
        super().__init__()
        d, dr = cfg["d_model"], cfg["dropout"]
        self.input_proj = nn.Linear(1, d)
        self.drop_in = nn.Dropout(dr)
        self.blocks = nn.ModuleList([RetNetBlock(d, cfg["n_heads"], cfg["ffn_dim"], dr)
                                     for _ in range(cfg["n_layers"])])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d * n_features, 256), nn.GELU(),
                                  nn.Dropout(dr), nn.Linear(256, n_outputs))

    def forward(self, x):
        x = self.drop_in(self.input_proj(x.unsqueeze(-1)))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x).reshape(x.size(0), -1))


def retnet_default_config():
    return {"d_model": 64, "n_heads": 4, "n_layers": 3, "ffn_dim": 128,
            "dropout": 0.15, "lr": 1e-3, "batch_size": 64}


def retnet_sample_config(rng):
    # search space of Table 5
    d = int(rng.choice([32, 48, 64, 96]))
    return {"d_model": d, "n_heads": int(rng.choice([2, 4])),
            "n_layers": int(rng.choice([2, 3, 4])),
            "ffn_dim": d * int(rng.choice([2, 3, 4])),
            "dropout": float(rng.choice([0.1, 0.15, 0.2, 0.25])),
            "lr": float(rng.choice([5e-4, 1e-3, 2e-3])),
            "batch_size": int(rng.choice([32, 64, 96]))}


# Model: joint RetNet
SIZE_UNIT = "parameters"


class RetNetJoint(NNModel):
    name, joint = "retnet_joint", True

    def build(self, n_features, n_outputs, cfg):
        return RetNetTabular(n_features, n_outputs, cfg)

    def default_config(self):
        return retnet_default_config()

    def sample_config(self, rng):
        return retnet_sample_config(rng)


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    run_model(RetNetJoint())
