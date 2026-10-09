# Closed-loop validation files — split 0 (seed 42), 180 held-out test designs
# Writes, for the SAME 180 test designs used in repeat 0 of the fair benchmark:
#   closedloop_split0_targets.csv                      original 19 design params + 28 specs
#   closedloop_split0_<model>_predicted_params.csv     predicted 19 design params (physical units)
# for model in {retnet_joint, mlp_joint, retnet_sep}, using the saved predictions


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV = "design_points_results_combined.csv"
PRED_DIR = PRED_DIR = "."
MODELS = ["retnet_joint", "mlp_joint", "retnet_sep"]
REPEAT, BASE_SEED = 0, 42
TARGETS = ["L1", "L3", "L5", "W1", "W3", "W5", "W6", "W7", "IBIAS", "Rz", "CC"]  # order used in the npz
DESIGN_COLS = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
               "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "IBIAS", "CC", "Rz"]
# topology constraints
COPY = {"L2": "L1", "L4": "L3", "L6": "L3", "L7": "L5", "L8": "L5",
        "W2": "W1", "W4": "W3", "W8": "W5"}

df = pd.read_csv(INPUT_CSV).iloc[:, :47]
df = df.dropna().reset_index(drop=True)

# the split used by the fair scripts (repeat 0)
seed = BASE_SEED + REPEAT
tr_full, te = train_test_split(np.arange(len(df)), test_size=0.20, random_state=seed)
assert len(te) == 180

test = df.iloc[te].copy()
test.insert(0, "csv_row_excel", te + 2)      # row number as seen in Excel (header = row 1)
test.insert(0, "dataset_index", te)          # 0-based index in the CSV
test.insert(0, "test_id", np.arange(len(te)))
test.to_csv("closedloop_split0_targets.csv", index=False)

for m in MODELS:
    path = os.path.join(PRED_DIR, f"{m}_r{REPEAT}.npz")
    log_pred = np.load(path)["test"]                     # (180, 11), log10, benchmark order
    assert log_pred.shape == (180, 11), log_pred.shape
    p = pd.DataFrame(10.0 ** log_pred, columns=TARGETS)
    for dst, src in COPY.items():
        p[dst] = p[src]
    p = p[DESIGN_COLS]
    p.insert(0, "csv_row_excel", te + 2); p.insert(0, "dataset_index", te); p.insert(0, "test_id", np.arange(len(te)))
    out = f"closedloop_split0_{m}_predicted_params.csv"
    p.to_csv(out, index=False)
    # sanity check: MAPE on the 11 modelled targets must match the benchmark's repeat-0 results
    true = df.iloc[te][TARGETS].values
    mape = np.mean(np.abs((true - p[TARGETS].values) / true)) * 100
    print(f"Wrote {out}   (check: mean MAPE over 11 targets = {mape:.2f}% — compare with repeat 0 in {m}_log.txt)")