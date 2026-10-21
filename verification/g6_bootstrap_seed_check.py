"""G6: are the R3 CI differences Monte-Carlo noise? (1) replicate the report's resampling stream
(numpy default_rng(22), one integers(0, n_seg, n_seg) draw per resample, segments in sorted order)
with this script's own weighted-MAE kernel -> should reproduce the printed CIs; (2) spread of the CI
bounds over 5 other seeds."""
import json, sys
import numpy as np, pandas as pd
sys.argv += [] if len(sys.argv) > 1 else []
PRED = sys.argv[1]; DATA = sys.argv[2]
df = pd.read_parquet(DATA + "/court_phase3.parquet", columns=["segment_id", "timestamp", "n_hist"])
df["timestamp"] = pd.to_datetime(df["timestamp"])
d = df.sort_values(["segment_id", "timestamp"], kind="mergesort")
d["j"] = d.groupby("segment_id").cumcount() + 1 - d["n_hist"].astype(int)
end = d.set_index(["segment_id", "j"])["timestamp"]

def ci(m, seed, n_boot=1000):
    segs, inv = np.unique(m["segment_id"].to_numpy(), return_inverse=True)
    ex = np.bincount(inv, np.abs(m.y_true - m.y_pred_x)); eb = np.bincount(inv, np.abs(m.y_true - m.y_pred_b)); c = np.bincount(inv)
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(segs), len(segs)), minlength=len(segs))
        out.append((w @ ex - w @ eb) / (w @ c))
    return np.nanpercentile(out, [2.5, 97.5])

res = []
for method, tier in (("single_agg", "tier1a"), ("prefix_agg", "tier1b")):
    b = pd.read_csv("%s/%s__court_p3_reg__%s__FS-B0__xgboost.csv.gz" % (PRED, tier, method))
    for fs in ("FS-M", "FS-MP", "FS-B0+P", "FS-B0+G", "FS-ALL"):
        x = pd.read_csv("%s/%s__court_p3_reg__%s__%s__xgboost.csv.gz" % (PRED, tier, method, fs))
        m = x.merge(b[["segment_id", "prefix_len", "y_pred"]], on=["segment_id", "prefix_len"], suffixes=("_x", "_b"))
        for restricted in (False, True):
            mm = m
            if restricted:
                ed = end.reindex(pd.MultiIndex.from_arrays([m.segment_id, m.prefix_len])).to_numpy()
                mm = m[ed <= np.datetime64("2016-12-31")]
            s22 = ci(mm, 22)
            others = np.array([ci(mm, s) for s in (1, 2, 3, 4, 5)])
            res.append({"row": "R3 %s %s%s" % (method, fs, " <=2016" if restricted else ""),
                        "seed22": [round(v, 1) for v in s22],
                        "other_seeds_lo_range": [round(others[:, 0].min(), 1), round(others[:, 0].max(), 1)],
                        "other_seeds_hi_range": [round(others[:, 1].min(), 1), round(others[:, 1].max(), 1)]})
            print(res[-1])
json.dump(res, open("verification/g6_bootstrap_seed_check.json", "w"), indent=1)
