import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


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

# Curated feature groups aligned to BASiC main CSV columns discovered from column_list.csv.
CURATED_FEATURE_PRIORITY = [
    # GPS / navigation
    "GPS_NSats", "GPS_HDop", "GPS_Lat", "GPS_Lng", "GPS_Alt", "GPS_Spd", "GPS_VZ", "GPS_Yaw",
    "GPA_HAcc", "GPA_VAcc", "GPA_SAcc", "POS_Lat", "POS_Lng", "POS_Alt",
    # IMU / acceleration / gyroscope
    "IMU_GyrX", "IMU_GyrY", "IMU_GyrZ", "IMU_AccX", "IMU_AccY", "IMU_AccZ", "IMU_EG", "IMU_EA",
    "VIBE_VibeX", "VIBE_VibeY", "VIBE_VibeZ",
    # Compass / heading
    "MAG_MagX", "MAG_MagY", "MAG_MagZ", "ATT_Yaw", "AHR2_Yaw", "RATE_Y", "RATE_YDes",
    # Barometer / altitude
    "BARO_Alt", "BARO_Press", "BARO_Temp", "CTUN_Alt", "CTUN_BAlt", "CTUN_DAlt", "CTUN_CRt",
    "TERR_CHeight",
    # RC / attitude control
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


class TabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(256, 128), dropout: float = 0.3, num_classes: int = 6):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)])
            last = h
        layers.append(nn.Linear(last, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        # Prefer the merged main CSV whose stem matches the folder name.
        if stem == folder:
            out.append(csv_path)
            continue
        # Fallback: include any CSV that contains Status when read later.
        if "small_data_analysis_pack" in str(csv_path):
            # avoid window helper CSVs when full dataset root is provided
            continue
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
    # BASiC processed data Time appears to be integer ticks; heuristic below handles us/ms/s scales.
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
        detected_minus_metadata_rows = None
        if detected_idx is not None and metadata_idx is not None:
            detected_minus_metadata_rows = int(detected_idx - metadata_idx)
        detected_minus_metadata_time = None
        if detected_time is not None and metadata_time is not None:
            detected_minus_metadata_time = float(detected_time - metadata_time)

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
            "detected_minus_metadata_rows": detected_minus_metadata_rows,
            "detected_minus_metadata_time": detected_minus_metadata_time,
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
        raise RuntimeError("No valid failure-flight samples were built. Check dataset root and Status column.")
    return samples, feature_names, anchor_report, (selected_feature_cols or [])


def build_augmented_training_samples(
    base_samples: Sequence[FlightSample],
    feature_cols: Sequence[str],
    stats_mode: str,
    window_sec: float,
    shifts_ms: Sequence[int],
) -> List[FlightSample]:
    out = list(base_samples)
    for s in base_samples:
        df = read_main_csv(Path(s.csv_path))
        if df is None:
            continue
        for ms in shifts_ms:
            offset = int(round(s.hz * ms / 1000.0))
            aug_idx = max(0, min(len(df) - 1, s.anchor_idx + offset))
            window_df = extract_window_by_anchor(df, aug_idx, s.hz, window_sec)
            vec, _ = build_stat_feature_vector(window_df, feature_cols, stats_mode=stats_mode)
            out.append(FlightSample(
                flight_id=f"{s.flight_id}_shift{ms}",
                flight_folder=s.flight_folder,
                label=s.label,
                csv_path=s.csv_path,
                anchor_idx=aug_idx,
                anchor_time=s.anchor_time,
                detected_idx=s.detected_idx,
                detected_time=s.detected_time,
                metadata_idx=s.metadata_idx,
                metadata_time=s.metadata_time,
                hz=s.hz,
                feature_vector=vec,
            ))
    return out


def prepare_arrays(samples: Sequence[FlightSample]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    x = np.stack([s.feature_vector for s in samples], axis=0)
    y = np.asarray([s.label for s in samples], dtype=np.int64)
    flight_ids = [s.flight_id for s in samples]
    flight_folders = [s.flight_folder for s in samples]
    return x, y, flight_ids, flight_folders


def train_one_split(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    loss_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dims: Tuple[int, int],
    dropout: float,
    seed: int,
) -> Dict:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    class_counts = np.bincount(train_y, minlength=6)
    class_weights = len(train_y) / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    ds_train = TabularDataset(train_x, train_y)
    ds_test = TabularDataset(test_x, test_y)

    sample_weights = np.asarray([class_weights[y] for y in train_y], dtype=np.float64)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    model = MLPClassifier(in_dim=train_x.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss(alpha=class_weights_t, gamma=2.0) if loss_name == "focal" else nn.CrossEntropyLoss(weight=class_weights_t)

    best_f1 = -1.0
    best_state = None
    patience = 20
    bad_rounds = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        preds = []
        gts = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(pred.tolist())
                gts.extend(yb.numpy().tolist())
        _, _, f1, _ = precision_recall_fscore_support(gts, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_rounds = 0
        else:
            bad_rounds += 1
            if bad_rounds >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    gts = []
    probs = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1)
            preds.extend(pred.tolist())
            gts.extend(yb.numpy().tolist())
            probs.append(prob)
    probs = np.concatenate(probs, axis=0)

    precision, recall, f1, _ = precision_recall_fscore_support(gts, preds, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(gts, preds)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(gts, preds, labels=list(range(6))).tolist(),
        "classification_report": classification_report(
            gts,
            preds,
            labels=list(range(6)),
            target_names=[IDX2LABEL[i] for i in range(6)],
            zero_division=0,
            output_dict=True,
        ),
        "probabilities": probs.tolist(),
        "predictions": preds,
        "ground_truth": gts,
    }


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
        train_samples = [samples[i] for i in train_idx]
        test_samples = [samples[i] for i in test_idx]

        if args.use_augmentation:
            train_samples = build_augmented_training_samples(
                base_samples=train_samples,
                feature_cols=selected_feature_cols,
                stats_mode=args.stats_mode,
                window_sec=args.window_sec,
                shifts_ms=args.augment_shifts_ms,
            )

        train_x, train_y, train_ids, _ = prepare_arrays(train_samples)
        test_x, test_y, test_ids, test_folders = prepare_arrays(test_samples)

        metrics = train_one_split(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            loss_name=args.loss,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
            seed=args.seed + split_id,
        )

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
        "feature_profile": args.feature_profile,
        "stats_mode": args.stats_mode,
        "window_sec": args.window_sec,
        "loss": args.loss,
        "test_size": args.test_size,
        "repeats": args.repeats,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "hidden_dims": args.hidden_dims,
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
    parser = argparse.ArgumentParser(description="Enhanced statistical feature + lightweight MLP classifier for BASiC Processed Data")
    parser.add_argument("--data-root", type=str, required=True, help="Root of BASiC Processed Data, e.g. datasets/BASiC/Processed Data")
    parser.add_argument("--metadata-csv", type=str, default="", help="Optional metadata.csv from small_data_analysis_pack")
    parser.add_argument("--output-dir", type=str, default="outputs/esf_csmlp_v2")
    parser.add_argument("--window-sec", type=float, default=1.0, help="Main setting: 1.0; comparison setting: 3.0")
    parser.add_argument("--feature-profile", type=str, default="curated", choices=["curated", "full_numeric"])
    parser.add_argument("--stats-mode", type=str, default="enhanced", choices=["enhanced", "mean_std"])
    parser.add_argument("--loss", type=str, default="weighted_ce", choices=["weighted_ce", "focal"])
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument("--augment-shifts-ms", nargs="+", type=int, default=[-150, -75, 75, 150])
    args = parser.parse_args()
    run_experiment(args)
