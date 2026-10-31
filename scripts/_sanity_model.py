import time, json, numpy as np, pandas as pd
from collision_risk.config import load_config
from collision_risk import data, features as FE, train as T, calibration as C
cfg = load_config(); cfg["model"]["max_rounds"] = 400; cfg["model"]["early_stopping_rounds"] = 50
labels, info = data.load_event_labels(cfg)
tr = pd.read_parquet("data/processed/train.parquet")
snap = FE.build_event_snapshots(data.apply_horizon(tr, 2.0), cfg)
snap = snap.merge(labels[["event_id", "final_tier", "is_trivial", "usable_for_training"]], on="event_id")
split = T.make_holdout_split(labels, cfg); print({k: v["n_events"] for k, v in split["counts"].items()}, "crit share", [round(v["critical_share"],3) for v in split["counts"].values()])
dev = snap[snap.event_id.isin(split["events"]["dev"])].reset_index(drop=True)
y = dev.final_tier.map(T.TIER_INDEX).to_numpy(); X = FE.feature_matrix(dev, "final_tier"); w = T.snapshot_weights(dev, y, cfg)
groups = dev.event_id.to_numpy(); eval_mask = dev.is_latest_snapshot.to_numpy() & ~dev.is_trivial.to_numpy()
params = {"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 100, "feature_fraction": 0.9, "lambda_l2": 1.0}
t0 = time.time(); oof, iters = T.grouped_oof_multiclass(X, y, groups, w, params, cfg, n_folds=3); print("oof 3-fold", round(time.time()-t0,1), "s iters", iters)
cf = T.carry_forward_probs(dev.tier_now_from_risk.map({0.0:"Safe",1.0:"Elevated",2.0:"Critical"}).to_numpy())
for name, P in (("carry_forward", cf), ("lgbm raw t=0.5", oof)):
    print(name, "real_forecast:", T.critical_f2_from_probs(P[eval_mask], y[eval_mask]), "macroF1", round(T.macro_f1_from_probs(P[eval_mask], y[eval_mask]),3))
cal = C.MulticlassIsotonicCalibrator().fit(oof[dev.is_latest_snapshot.to_numpy()], y[dev.is_latest_snapshot.to_numpy()])
Pc = cal.predict_proba(oof); thr = C.tune_decision_thresholds(Pc[eval_mask], y[eval_mask]); print("thresholds", thr)
pred = C.apply_decision_rule(Pc[eval_mask], thr["t_crit"], thr["t_elev"]); yt = y[eval_mask]
from sklearn.metrics import confusion_matrix, f1_score
print(confusion_matrix(yt, pd.Series(pred).map(T.TIER_INDEX).to_numpy(), labels=[0,1,2])); print("macroF1", round(f1_score(yt, pd.Series(pred).map(T.TIER_INDEX), average="macro"),3))
# latest-only fit comparison
t0 = time.time(); oof2, _ = T.grouped_oof_multiclass(X, y, groups, w, params, cfg, n_folds=3, fit_mask=dev.is_latest_snapshot.to_numpy()); print("latest-only", round(time.time()-t0,1), "s", T.critical_f2_from_probs(oof2[eval_mask], y[eval_mask]))
lac = C.LACPredictionSets(0.9).fit(Pc[eval_mask], yt); sets = lac.predict_sets(Pc[eval_mask]); print("LAC", lac.to_dict(), C.prediction_set_metrics(sets, np.array(T.TIERS)[yt]))
