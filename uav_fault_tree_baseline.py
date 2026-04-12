import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

LABEL_MAP = {
    "gps": 0,
    "rc": 1,
    "remote control": 1,
    "accelerometer": 2,
    "accelero": 2,
    "gyro": 3,
    "gyroscope": 3,
    "compass": 4,
    "mag": 4,
    "barometer": 5,
    "baro": 5,
}

IDX2LABEL = {
    0: "GPS",
    1: "RC",
    2: "Accelerometer",
    3: "Gyroscope",
    4: "Compass",
    5: "Barometer",
}

IGNORE_COLS = {"Time", "Status", "time", "timestamp", "Timestamp"}

CURATED_FEATURE_PRIORITY = [
    "GPS_NSats", "GPS_HDop", "GPS_Lat", "GPS_Lng", "GPS_Alt", "GPS_Spd", "GPS_VZ", "GPS_Yaw",
    "GPA_HAcc", "GPA_VAcc", "GPA_SAcc", "POS_Lat", "POS_Lng", "POS_Alt",
    "IMU_GyrX", "IMU_GyrY", "IMU_GyrZ", "IMU_AccX", "IMU_AccY", "IMU_AccZ", "IMU_EG", "IMU_EA",
    "VIBE_VibeX", "VIBE_VibeY", "VIBE_VibeZ",
    "MAG_MagX", "MAG_MagY", "MAG_MagZ", "ATT_Yaw", "AHR2_Yaw", "RATE_Y", "RATE_YDes",
    "BARO_Alt", "BARO_Press", "BARO_Temp", "CTUN_Alt", "CTUN_BAlt", "CTUN_DAlt", "CTUN_CRt",
    "TERR_CHeight",
    "ATT_DesRoll", "ATT_Roll", "ATT_DesPitch", "ATT_Pitch", "CTUN_ThO", "RATE_RDes", "RATE_R",
    "RATE_PDes", "RATE_P", "MAV_txp", "MAV_rxp",
]

ENHANCED_STATS = [
    "mean", "std", "max", "min", "range", "rms", "median", "q25", "q75",
    "abs_mean", "diff_mean", "diff_std", "slope",
]

MEAN_STD_STATS = ["mean", "std"]


@dataclass
class FlightSample:
    flight_id: str
    flight_folder: str
    label: int
    csv_path: str
    anchor_idx: int
    anchor_time: Optional[float]
    detected_idx: Optional[int]
    detected_time: Optional[float]
    metadata_idx: Optional[int]
    metadata_time: Optional[float]
    hz: float
    feature_vector: np.ndarray


def infer_label_from_name(name: str) -> Optional[int]:
    low = name.lower()
    if "no failure" in low:
        return None
    for k, v in LABEL_MAP.items():
        if k in low:
            return v
    return None


def load_metadata(metadata_csv: Optional[Path]) -> Dict[str, Dict[str, Optional[float]]]:
    if metadata_csv is None or not metadata_csv.exists():
        return {}
    meta = pd.read_csv(metadata_csv)
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for _, row in meta.iterrows():
        folder = str(row.get("sample_folder", "")).strip()
        if not folder:
            continue
        out[folder] = {
            "fault_row_index": None if pd.isna(row.get("fault_row_index")) else int(row.get("fault_row_index")),
            "fault_time": None if pd.isna(row.get("fault_time")) else float(row.get("fault_time")),
        }
    return out


def discover_main_csvs(root: Path) -> List[Path]:
    out = []
    for csv_path in root.rglob("*.csv"):
        if not csv_path.is_file():
            continue
        folder = csv_path.parent.name
        stem = csv_path.stem
        if stem == folder:
            out.append(csv_path)
    return sorted(set(out))


def read_main_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "Status" not in df.columns or "Time" not in df.columns:
        return None
    return df


def to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def detect_status_transition(df: pd.DataFrame) -> Tuple[Optional[int], Optional[float]]:
    status = to_numeric_series(df, "Status").fillna(0).astype(float).values
    time_vals = to_numeric_series(df, "Time").values
    for i in range(1, len(status)):
        if status[i - 1] <= 0 and status[i] > 0:
            t = time_vals[i] if i < len(time_vals) and np.isfinite(time_vals[i]) else np.nan
            return i, None if not np.isfinite(t) else float(t)
    pos = np.where(status > 0)[0]
    if len(pos):
        i = int(pos[0])
        t = time_vals[i] if i < len(time_vals) and np.isfinite(time_vals[i]) else np.nan
        return i, None if not np.isfinite(t) else float(t)
    return None, None


def estimate_hz(df: pd.DataFrame, default_hz: float = 400.0) -> float:
    t = to_numeric_series(df, "Time").dropna().values
    if len(t) < 10:
        return default_hz
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return default_hz
    median_dt = float(np.median(dt))
    if median_dt > 1000:
        hz = 1e6 / median_dt
    elif median_dt > 1:
        hz = 1e3 / median_dt
    else:
        hz = 1.0 / median_dt
    if not np.isfinite(hz) or hz <= 0:
        return default_hz
    return float(hz)


def nearest_index_by_time(df: pd.DataFrame, target_time: float) -> Optional[int]:
    t = to_numeric_series(df, "Time").values.astype(float)
    if len(t) == 0 or not np.isfinite(target_time):
        return None
    mask = np.isfinite(t)
    if not mask.any():
        return None
    idx = int(np.argmin(np.abs(t[mask] - target_time)))
    valid_positions = np.where(mask)[0]
    return int(valid_positions[idx])


def select_feature_columns(df: pd.DataFrame, feature_profile: str) -> List[str]:
    if feature_profile == "curated":
        return [c for c in CURATED_FEATURE_PRIORITY if c in df.columns]

    numeric_cols = []
    for c in df.columns:
        if c in IGNORE_COLS:
            continue
        s = to_numeric_series(df, c)
        if s.notna().mean() >= 0.8:
            numeric_cols.append(c)
    return numeric_cols


def _safe_stats(x: np.ndarray, stats_mode: str) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    wanted = MEAN_STD_STATS if stats_mode == "mean_std" else ENHANCED_STATS
    if len(x) == 0:
        return {k: 0.0 for k in wanted}
    diff = np.diff(x) if len(x) > 1 else np.array([0.0])
    slope = (x[-1] - x[0]) / max(len(x) - 1, 1)
    stat_map = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
        "range": float(np.max(x) - np.min(x)),
        "rms": float(np.sqrt(np.mean(np.square(x)))),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "abs_mean": float(np.mean(np.abs(x))),
        "diff_mean": float(np.mean(diff)),
        "diff_std": float(np.std(diff)),
        "slope": float(slope),
    }
    return {k: stat_map[k] for k in wanted}


def extract_window_by_anchor(df: pd.DataFrame, anchor_idx: int, hz: float, window_sec: float) -> pd.DataFrame:
    half = int(round(hz * window_sec / 2.0))
    half = max(1, half)
    start = max(0, anchor_idx - half)
    end = min(len(df), anchor_idx + half)
    return df.iloc[start:end].copy()


def build_stat_feature_vector(window_df: pd.DataFrame, feature_cols: Sequence[str], stats_mode: str) -> Tuple[np.ndarray, List[str]]:
    feats: List[float] = []
    names: List[str] = []
    for col in feature_cols:
        x = to_numeric_series(window_df, col).values.astype(float)
        stats = _safe_stats(x, stats_mode=stats_mode)
        for k, v in stats.items():
            feats.append(v)
            names.append(f"{col}__{k}")
    return np.asarray(feats, dtype=np.float32), names


def choose_anchor(df: pd.DataFrame, folder_name: str, metadata: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    detected_idx, detected_time = detect_status_transition(df)
    meta = metadata.get(folder_name, {})
    metadata_idx = meta.get("fault_row_index")
    metadata_time = meta.get("fault_time")

    anchor_time = metadata_time if metadata_time is not None else detected_time
    anchor_idx = None
    if anchor_time is not None:
        anchor_idx = nearest_index_by_time(df, anchor_time)
    if anchor_idx is None:
        anchor_idx = metadata_idx if metadata_idx is not None else detected_idx
    if anchor_idx is None:
        raise RuntimeError(f"Cannot locate fault anchor for {folder_name}")

    return {
        "anchor_idx": int(anchor_idx),
        "anchor_time": None if anchor_time is None else float(anchor_time),
        "detected_idx": None if detected_idx is None else int(detected_idx),
        "detected_time": None if detected_time is None else float(detected_time),
        "metadata_idx": None if metadata_idx is None else int(metadata_idx),
        "metadata_time": None if metadata_time is None else float(metadata_time),
    }


def build_samples(
    root: Path,
    metadata_csv: Optional[Path],
    window_sec: float,
    feature_profile: str,
    stats_mode: str,
) -> Tuple[List[FlightSample], List[str], List[Dict[str, Optional[float]]], List[str]]:
    metadata = load_metadata(metadata_csv)
    main_csvs = discover_main_csvs(root)
    if not main_csvs:
        raise FileNotFoundError(f"No main CSV files found under {root}")

    samples: List[FlightSample] = []
    feature_names: List[str] = []
    selected_feature_cols: Optional[List[str]] = None
    anchor_report: List[Dict[str, Optional[float]]] = []

    for csv_path in main_csvs:
        folder_name = csv_path.parent.name
        label = infer_label_from_name(folder_name)
        if label is None:
            continue

        df = read_main_csv(csv_path)
        if df is None:
            continue

        hz = estimate_hz(df)
        anchor = choose_anchor(df, folder_name, metadata)

        if selected_feature_cols is None:
            selected_feature_cols = select_feature_columns(df, feature_profile=feature_profile)
        usable_cols = [c for c in selected_feature_cols if c in df.columns]
        if not usable_cols:
            continue

        window_df = extract_window_by_anchor(df, anchor_idx=anchor["anchor_idx"], hz=hz, window_sec=window_sec)
        vec, feature_names = build_stat_feature_vector(window_df, usable_cols, stats_mode=stats_mode)

        detected_idx = anchor["detected_idx"]
        detected_time = anchor["detected_time"]
        metadata_idx = anchor["metadata_idx"]
        metadata_time = anchor["metadata_time"]
        anchor_report.append({
            "flight_folder": folder_name,
            "label": IDX2LABEL[label],
            "csv_path": str(csv_path),
            "hz": hz,
            "anchor_idx": anchor["anchor_idx"],
            "anchor_time": anchor["anchor_time"],
            "detected_idx": detected_idx,
            "detected_time": detected_time,
            "metadata_idx": metadata_idx,
            "metadata_time": metadata_time,
            "window_sec": window_sec,
            "window_rows": len(window_df),
        })

        samples.append(FlightSample(
            flight_id=csv_path.stem,
            flight_folder=folder_name,
            label=label,
            csv_path=str(csv_path),
            anchor_idx=anchor["anchor_idx"],
            anchor_time=anchor["anchor_time"],
            detected_idx=detected_idx,
            detected_time=detected_time,
            metadata_idx=metadata_idx,
            metadata_time=metadata_time,
            hz=hz,
            feature_vector=vec,
        ))

    if not samples:
        raise RuntimeError("No valid failure-flight samples were built.")
    return samples, feature_names, anchor_report, (selected_feature_cols or [])


def prepare_arrays(samples: Sequence[FlightSample]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    x = np.stack([s.feature_vector for s in samples], axis=0)
    y = np.asarray([s.label for s in samples], dtype=np.int64)
    flight_ids = [s.flight_id for s in samples]
    flight_folders = [s.flight_folder for s in samples]
    return x, y, flight_ids, flight_folders


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    class_counts = np.bincount(y, minlength=6)
    class_weights = len(y) / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    return np.asarray([class_weights[label] for label in y], dtype=np.float64)


def make_model(args, seed: int):
    if args.model == "rf":
        return RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            max_features=args.rf_max_features,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=seed,
        )

    if args.model == "xgb":
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise RuntimeError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            ) from e
        return XGBClassifier(
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            objective="multi:softprob",
            num_class=6,
            eval_metric="mlogloss",
            reg_lambda=args.xgb_reg_lambda,
            min_child_weight=args.xgb_min_child_weight,
            random_state=seed,
            n_jobs=args.n_jobs,
            tree_method=args.xgb_tree_method,
        )

    raise ValueError(f"Unsupported model: {args.model}")


def fit_predict(args, train_x, train_y, test_x, test_y, seed: int) -> Dict:
    model = make_model(args, seed)
    if args.model == "xgb":
        sample_weight = compute_sample_weights(train_y)
        model.fit(train_x, train_y, sample_weight=sample_weight)
    else:
        model.fit(train_x, train_y)

    preds = model.predict(test_x)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(test_x)
    else:
        probs = None

    precision, recall, f1, _ = precision_recall_fscore_support(test_y, preds, average="macro", zero_division=0)
    out = {
        "accuracy": float(accuracy_score(test_y, preds)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(test_y, preds, labels=list(range(6))).tolist(),
        "classification_report": classification_report(
            test_y,
            preds,
            labels=list(range(6)),
            target_names=[IDX2LABEL[i] for i in range(6)],
            zero_division=0,
            output_dict=True,
        ),
        "predictions": preds.tolist(),
        "ground_truth": test_y.tolist(),
    }
    if probs is not None:
        out["probabilities"] = probs.tolist()
    return out


def aggregate_results(split_metrics: List[Dict]) -> Dict:
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    return {
        "metrics_mean": {k: float(np.mean([m[k] for m in split_metrics])) for k in keys},
        "metrics_std": {k: float(np.std([m[k] for m in split_metrics])) for k in keys},
    }


def run_experiment(args):
    root = Path(args.data_root)
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else None

    samples, feature_names, anchor_report, selected_feature_cols = build_samples(
        root=root,
        metadata_csv=metadata_csv,
        window_sec=args.window_sec,
        feature_profile=args.feature_profile,
        stats_mode=args.stats_mode,
    )

    x, y, ids, folders = prepare_arrays(samples)
    splitter = StratifiedShuffleSplit(n_splits=args.repeats, test_size=args.test_size, random_state=args.seed)

    split_summaries: List[Dict] = []
    rows_for_metrics: List[Dict] = []

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        train_x = x[train_idx]
        train_y = y[train_idx]
        test_x = x[test_idx]
        test_y = y[test_idx]
        test_ids = [ids[i] for i in test_idx]
        test_folders = [folders[i] for i in test_idx]

        metrics = fit_predict(args, train_x, train_y, test_x, test_y, seed=args.seed + split_id)
        metrics["split_id"] = split_id
        metrics["test_flight_ids"] = test_ids
        metrics["test_flight_folders"] = test_folders
        split_summaries.append(metrics)
        rows_for_metrics.append({
            "split_id": split_id,
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
        })

    aggregate = aggregate_results(split_summaries)
    summary = {
        "task": "BASiC stage-2 six-class sensor failure classification",
        "n_failure_flights": int(len(samples)),
        "classes": {str(i): IDX2LABEL[i] for i in range(6)},
        "model": args.model,
        "feature_profile": args.feature_profile,
        "stats_mode": args.stats_mode,
        "window_sec": args.window_sec,
        "test_size": args.test_size,
        "repeats": args.repeats,
        "feature_dim": int(x.shape[1]),
        "num_selected_columns": int(len(selected_feature_cols)),
        **aggregate,
        "split_results": split_summaries,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(anchor_report).to_csv(out_dir / "sample_anchor_check.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_for_metrics).to_csv(out_dir / "split_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature_name": feature_names}).to_csv(out_dir / "feature_names.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "features_used.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected_feature_columns": selected_feature_cols,
            "feature_profile": args.feature_profile,
            "stats_mode": args.stats_mode,
        }, f, ensure_ascii=False, indent=2)

    mean_cm = np.mean([np.asarray(s["confusion_matrix"], dtype=float) for s in split_summaries], axis=0)
    cm_df = pd.DataFrame(mean_cm, index=[IDX2LABEL[i] for i in range(6)], columns=[IDX2LABEL[i] for i in range(6)])
    cm_df.to_csv(out_dir / "confusion_matrix_mean.csv", encoding="utf-8-sig")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(json.dumps(summary["metrics_mean"], ensure_ascii=False, indent=2))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tree-model baselines for BASiC stage-2 classification")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--metadata-csv", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--feature-profile", type=str, default="curated", choices=["curated", "full_numeric"])
    parser.add_argument("--stats-mode", type=str, default="enhanced", choices=["enhanced", "mean_std"])
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "xgb"])
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument("--rf-n-estimators", type=int, default=400)
    parser.add_argument("--rf-max-depth", type=int, default=8)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--rf-max-features", type=str, default="sqrt")

    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    parser.add_argument("--xgb-tree-method", type=str, default="hist")

    args = parser.parse_args()
    run_experiment(args)
