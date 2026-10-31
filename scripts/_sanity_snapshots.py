import time, numpy as np, pandas as pd
from collision_risk.config import load_config
from collision_risk import data, features as FE
cfg = load_config()
t0 = time.time()
labels, info = data.load_event_labels(cfg)
print({k: v for k, v in info.items() if k != "workbook_path"})
tr = pd.read_parquet("data/processed/train.parquet")
h = data.apply_horizon(tr, cfg["horizon"]["min_days"])
print("horizon rows", len(h), "events", h.event_id.nunique(), "min t", h.time_to_tca.min())
snap = FE.build_event_snapshots(h, cfg)
print("snapshots built in", round(time.time()-t0,1), "s; shape", snap.shape, "latest rows", snap.is_latest_snapshot.sum())
assert (snap.row_id.to_numpy() == h.row_id.to_numpy()).all()
lat = snap[snap.is_latest_snapshot].merge(labels[["event_id","final_tier","is_trivial","usable_for_training"]], on="event_id")
assert lat.usable_for_training.all() and len(lat) == info["n_usable"]
sev = {0.0:"Safe",1.0:"Elevated",2.0:"Critical"}
lat["cf"] = lat.tier_now_from_risk.map(sev)
def m(d):
    tp = ((d.cf=="Critical")&(d.final_tier=="Critical")).sum(); fp=((d.cf=="Critical")&(d.final_tier!="Critical")).sum(); fn=((d.cf!="Critical")&(d.final_tier=="Critical")).sum()
    from sklearn.metrics import f1_score
    return dict(n=len(d), recall=round(tp/(tp+fn),3), precision=round(tp/(tp+fp),3), macro_f1=round(f1_score(d.final_tier, d.cf, average="macro"),3))
print("carry_forward_risk all_usable:", m(lat)); print("carry_forward_risk real_forecast:", m(lat[~lat.is_trivial])); print("trivial:", m(lat[lat.is_trivial]))
X = FE.feature_matrix(snap, "final_tier"); print(X.shape, "nan frac slopes", X[["slope_log_miss","slope_risk","volatility_mre"]].isna().mean().round(3).to_dict())
one = snap[snap.event_id==snap.event_id.iloc[0]].sort_values("time_to_tca", ascending=False)[["time_to_tca","n_cdms_2d","span_days","d_log_miss_first","slope_log_miss","slope_risk","volatility_risk","n_up_risk","n_down_risk","tier_now_from_risk","is_latest_snapshot"]]
print(one.to_string())
