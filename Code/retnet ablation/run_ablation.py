import argparse
import importlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ALL_VARIANTS = ["full", "no_blocks", "no_retention", "bidirectional", "no_decay",
                "shuffled:0", "shuffled:1", "shuffled:2", "shuffled:3", "shuffled:4"]
OUT = "ablation_results"


def find_run_info(form):
    name = f"retnet_{form}_run_info.csv"
    for p in (os.path.join("fair_results_origmask", name), name):
        if os.path.exists(p):
            return p


def make_variant_classes(rj):

    class MaskedRetention(rj.SimpleRetention):
        mode = "causal"

        def forward(self, X):
            B, N, _ = X.shape
            Q, K, V = self.W_Q(X), self.W_K(X), self.W_V(X)
            idx = torch.arange(N, device=X.device, dtype=X.dtype)
            diff = idx.unsqueeze(1) - idx.unsqueeze(0)
            if self.mode == "bidirectional":
                D = self.gamma ** diff.abs()
            elif self.mode == "no_decay":
                D = torch.ones_like(diff)
            else:
                D = torch.tril(self.gamma ** diff)
            scores = torch.bmm(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)
            return torch.bmm(scores * D.unsqueeze(0), V)

    class FFNOnlyBlock(nn.Module):
        #RetNetBlock without the retention sub-layer: no mixing between tokens.

        def __init__(self, d_model, ffn_dim, dropout):
            super().__init__()
            self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
            self.norm2 = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)

        def forward(self, X):
            return X + self.drop(self.ffn(self.norm2(X)))

    def build_variant(variant, n_features, n_outputs, cfg):
        cfg = dict(cfg)
        if variant == "no_blocks":
            cfg["n_layers"] = 0
        net = rj.RetNetTabular(n_features, n_outputs, cfg)
        if variant == "no_retention":
            net.blocks = nn.ModuleList([FFNOnlyBlock(cfg["d_model"], cfg["ffn_dim"], cfg["dropout"])
                                        for _ in range(cfg["n_layers"])])
        if variant in ("bidirectional", "no_decay"):
            for blk in net.blocks:
                for r in blk.msr.retentions:
                    r.__class__ = MaskedRetention
                    r.mode = variant
        return net

    class Variant(rj.NNModel):
        def __init__(self, variant, joint):
            self.variant, self.joint = variant, joint

        def build(self, n_features, n_outputs, cfg):
            return build_variant(self.variant, n_features, n_outputs, cfg)

    return Variant


def run(form, variants, repeats, smoke=False):
    rj = importlib.import_module(f"retnet_{form}_origmask")
    if smoke:  # quick check: 1 epoch per run
        rj.NN_FINAL_EPOCHS, rj.NN_FINAL_PATIENCE = 1, 1
    Variant = make_variant_classes(rj)
    info = pd.read_csv(find_run_info(form)).set_index("repeat")
    out = OUT + ("_smoke" if smoke else "")
    os.makedirs(out, exist_ok=True)
    print(f"device: {rj.DEVICE}   form: {form}   variants: {variants}   splits: {repeats}")

    X0, Y, _, _ = rj.load_data()
    t_all = time.time()
    for variant_name in variants:
        base, _, perm_seed = variant_name.partition(":")
        raw_path = os.path.join(out, f"{form}_{variant_name.replace(':', '_')}_raw.csv")
        done = set(pd.read_csv(raw_path)["repeat"]) if os.path.exists(raw_path) else set()
        X = X0
        if base == "shuffled":  # reorder the 34 tokens = reorder the input columns
            perm = np.random.default_rng(int(perm_seed)).permutation(X0.shape[1])
            X = X0.iloc[:, perm]
            base = "full"
        model = Variant(base, joint=(form == "joint"))

        for r in repeats:
            if r in done:
                print(f"{variant_name:14s} split {r}: done, skipping")
                continue
            tr, va, te = rj.make_split(len(X), r)
            Xs, Ys, sy = rj.preprocess(X, Y, tr)
            seed = rj.BASE_SEED + r
            cfg = json.loads(info.loc[r, "config"])
            t0 = time.time()
            pv, pt_all, size = model.fit_predict(cfg, Xs[tr], Ys[tr], Xs[va], Ys[va],
                                                 np.vstack([Xs[tr], Xs[te]]), seed, trial=False)
            pt_all = sy.inverse_transform(pt_all)
            ptr, pte = pt_all[:len(tr)], pt_all[len(tr):]
            df = pd.DataFrame(rj.all_metrics(Y[te], pte))
            df.insert(0, "variant", variant_name)
            df.insert(0, "repeat", r)
            df["n_params"] = size
            df["train_mean_R2"] = rj.mean_r2(Y[tr], ptr)
            df["fit_seconds"] = round(time.time() - t0, 1)
            df.to_csv(raw_path, mode="a", header=not os.path.exists(raw_path), index=False)
            print(f"{variant_name:14s} split {r}: MAPE {df.MAPE.mean():.2f}%  R2 {df.R2.mean():.4f}  "
                  f"params {size:,}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", choices=["joint", "sep"], default="joint")
    ap.add_argument("--variants", nargs="+", default=ALL_VARIANTS)
    ap.add_argument("--repeats", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--smoke", action="store_true", help="1 epoch per run, to check the code")
    a = ap.parse_args()
    run(a.form, a.variants, a.repeats, a.smoke)
