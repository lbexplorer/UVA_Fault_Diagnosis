import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

IGNORE_COLS = {"Time", "Status", "time", "timestamp", "Timestamp"}

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

IDX2LABEL = {0: "NoFailure", 1: "Failure"}


@dataclass
class FlightSample:
    flight_id: str
    flight_folder: str
    label: int
    csv_path: str
    anchor_idx: int
    anchor_time: Optional[float]
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


class MLPBinary(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(256, 128), dropout: float = 0.3, num_classes: int = 2):
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
        if "small_data_analysis_pack" in str(csv_path):
            continue
        folder = csv_path.parent.name
        if csv_path.stem == folder:
            out.append(csv_path)
    return sorted(set(out))


def read_main_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "Time" not in df.columns or "Status" not in df.columns:
        return None
    return df


def to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


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


def infer_binary_label(name: str) -> int:
    return 0 if "no failure" in name.lower() else 1


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


def extract_centered_window(df: pd.DataFrame, anchor_idx: int, hz: float, window_sec: float) -> pd.DataFrame:
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


def choose_failure_anchor(df: pd.DataFrame, folder_name: str, metadata: Dict[str, Dict[str, Optional[float]]]) -> Tuple[int, Optional[float]]:
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
        raise RuntimeError(f"Cannot locate failure anchor for {folder_name}")
    return int(anchor_idx), None if anchor_time is None else float(anchor_time)


def choose_normal_anchor(df: pd.DataFrame, mode: str = "middle") -> Tuple[int, Optional[float]]:
    n = len(df)
    if n == 0:
        raise RuntimeError("Empty No Failure dataframe")
    if mode == "middle":
        idx = n // 2
    elif mode == "q1":
        idx = n // 4
    elif mode == "q3":
        idx = (3 * n) // 4
    else:
        idx = n // 2
    tvals = to_numeric_series(df, "Time").values
    t = tvals[idx] if idx < len(tvals) and np.isfinite(tvals[idx]) else np.nan
    return idx, None if not np.isfinite(t) else float(t)


def build_samples(
    root: Path,
    metadata_csv: Optional[Path],
    window_sec: float,
    feature_profile: str,
    stats_mode: str,
    no_failure_anchor_mode: str,
) -> Tuple[List[FlightSample], List[str], List[Dict], List[str]]:
    metadata = load_metadata(metadata_csv)
    main_csvs = discover_main_csvs(root)
    if not main_csvs:
        raise FileNotFoundError(f"No main CSV files found under {root}")

    samples: List[FlightSample] = []
    feature_names: List[str] = []
    selected_feature_cols: Optional[List[str]] = None
    anchor_report: List[Dict] = []

    for csv_path in main_csvs:
        folder_name = csv_path.parent.name
        label = infer_binary_label(folder_name)
        df = read_main_csv(csv_path)
        if df is None:
            continue
        hz = estimate_hz(df)
        if selected_feature_cols is None:
            selected_feature_cols = select_feature_columns(df, feature_profile)
        usable_cols = [c for c in selected_feature_cols if c in df.columns]
        if not usable_cols:
            continue

        if label == 1:
            anchor_idx, anchor_time = choose_failure_anchor(df, folder_name, metadata)
            anchor_source = "failure_status_or_metadata"
        else:
            anchor_idx, anchor_time = choose_normal_anchor(df, no_failure_anchor_mode)
            anchor_source = f"no_failure_{no_failure_anchor_mode}"

        window_df = extract_centered_window(df, anchor_idx, hz, window_sec)
        vec, feature_names = build_stat_feature_vector(window_df, usable_cols, stats_mode)
        samples.append(FlightSample(
            flight_id=csv_path.stem,
            flight_folder=folder_name,
            label=label,
            csv_path=str(csv_path),
            anchor_idx=anchor_idx,
            anchor_time=anchor_time,
            hz=hz,
            feature_vector=vec,
        ))
        anchor_report.append({
            "flight_folder": folder_name,
            "label": IDX2LABEL[label],
            "csv_path": str(csv_path),
            "hz": hz,
            "anchor_idx": anchor_idx,
            "anchor_time": anchor_time,
            "anchor_source": anchor_source,
            "window_sec": window_sec,
            "window_rows": len(window_df),
        })

    if not samples:
        raise RuntimeError("No valid binary samples were built.")
    return samples, feature_names, anchor_report, (selected_feature_cols or [])


def build_augmented_training_samples(
    base_samples: Sequence[FlightSample],
    feature_cols: Sequence[str],
    stats_mode: str,
    window_sec: float,
    shifts_ms: Sequence[int],
    normal_extra_modes: Sequence[str],
) -> List[FlightSample]:
    out = list(base_samples)
    for s in base_samples:
        df = read_main_csv(Path(s.csv_path))
        if df is None:
            continue
        if s.label == 1:
            for ms in shifts_ms:
                offset = int(round(s.hz * ms / 1000.0))
                aug_idx = max(0, min(len(df) - 1, s.anchor_idx + offset))
                window_df = extract_centered_window(df, aug_idx, s.hz, window_sec)
                vec, _ = build_stat_feature_vector(window_df, feature_cols, stats_mode)
                out.append(FlightSample(
                    flight_id=f"{s.flight_id}_shift{ms}",
                    flight_folder=s.flight_folder,
                    label=s.label,
                    csv_path=s.csv_path,
                    anchor_idx=aug_idx,
                    anchor_time=s.anchor_time,
                    hz=s.hz,
                    feature_vector=vec,
                ))
        else:
            for mode in normal_extra_modes:
                aug_idx, aug_time = choose_normal_anchor(df, mode)
                window_df = extract_centered_window(df, aug_idx, s.hz, window_sec)
                vec, _ = build_stat_feature_vector(window_df, feature_cols, stats_mode)
                out.append(FlightSample(
                    flight_id=f"{s.flight_id}_{mode}",
                    flight_folder=s.flight_folder,
                    label=s.label,
                    csv_path=s.csv_path,
                    anchor_idx=aug_idx,
                    anchor_time=aug_time,
                    hz=s.hz,
                    feature_vector=vec,
                ))
    return out


def prepare_arrays(samples: Sequence[FlightSample]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    x = np.stack([s.feature_vector for s in samples], axis=0)
    y = np.asarray([s.label for s in samples], dtype=np.int64)
    ids = [s.flight_id for s in samples]
    folders = [s.flight_folder for s in samples]
    return x, y, ids, folders


def compute_binary_metrics(y_true, y_pred, y_prob) -> Dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[IDX2LABEL[0], IDX2LABEL[1]],
            zero_division=0,
            output_dict=True,
        ),
        "predictions": [int(v) for v in y_pred],
        "ground_truth": [int(v) for v in y_true],
        "probabilities": [float(v) for v in y_prob],
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = None
    return out


def train_mlp_binary(
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

    class_counts = np.bincount(train_y, minlength=2)
    class_weights = len(train_y) / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    ds_train = TabularDataset(train_x, train_y)
    ds_test = TabularDataset(test_x, test_y)
    sample_weights = np.asarray([class_weights[y] for y in train_y], dtype=np.float64)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    model = MLPBinary(in_dim=train_x.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
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
        probs = []
        gts = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                pred = (prob >= 0.5).astype(int)
                preds.extend(pred.tolist())
                probs.extend(prob.tolist())
                gts.extend(yb.numpy().tolist())
        m = compute_binary_metrics(gts, preds, probs)
        score = m["f1"]
        if score > best_f1:
            best_f1 = score
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
    probs = []
    gts = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            preds.extend(pred.tolist())
            probs.extend(prob.tolist())
            gts.extend(yb.numpy().tolist())
    return compute_binary_metrics(gts, preds, probs)


def train_tree_binary(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    model_name: str,
    seed: int,
) -> Dict:
    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(train_x, train_y)
        prob = model.predict_proba(test_x)[:, 1]
    elif model_name == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Please run: pip install xgboost")
        pos = int(np.sum(train_y == 1))
        neg = int(np.sum(train_y == 0))
        scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(train_x, train_y)
        prob = model.predict_proba(test_x)[:, 1]
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    pred = (prob >= 0.5).astype(int)
    return compute_binary_metrics(test_y, pred, prob)


def aggregate_results(split_metrics: List[Dict]) -> Dict:
    keys = ["accuracy", "precision", "recall", "f1"]
    out = {
        "metrics_mean": {k: float(np.mean([m[k] for m in split_metrics])) for k in keys},
        "metrics_std": {k: float(np.std([m[k] for m in split_metrics])) for k in keys},
    }
    aucs = [m["roc_auc"] for m in split_metrics if m.get("roc_auc") is not None]
    out["metrics_mean"]["roc_auc"] = float(np.mean(aucs)) if aucs else None
    out["metrics_std"]["roc_auc"] = float(np.std(aucs)) if aucs else None
    return out


def run_experiment(args):
    root = Path(args.data_root)
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else None

    samples, feature_names, anchor_report, selected_feature_cols = build_samples(
        root=root,
        metadata_csv=metadata_csv,
        window_sec=args.window_sec,
        feature_profile=args.feature_profile,
        stats_mode=args.stats_mode,
        no_failure_anchor_mode=args.no_failure_anchor_mode,
    )

    x, y, ids, folders = prepare_arrays(samples)
    splitter = StratifiedShuffleSplit(n_splits=args.repeats, test_size=args.test_size, random_state=args.seed)

    split_summaries: List[Dict] = []
    rows_for_metrics: List[Dict] = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        train_samples = [samples[i] for i in train_idx]
        test_samples = [samples[i] for i in test_idx]

        if args.use_augmentation and args.model == "mlp":
            train_samples = build_augmented_training_samples(
                base_samples=train_samples,
                feature_cols=selected_feature_cols,
                stats_mode=args.stats_mode,
                window_sec=args.window_sec,
                shifts_ms=args.augment_shifts_ms,
                normal_extra_modes=args.normal_extra_modes,
            )

        train_x, train_y, _, _ = prepare_arrays(train_samples)
        test_x, test_y, test_ids, test_folders = prepare_arrays(test_samples)

        if args.model == "mlp":
            metrics = train_mlp_binary(
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
        else:
            metrics = train_tree_binary(
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y,
                model_name=args.model,
                seed=args.seed + split_id,
            )

        metrics["split_id"] = split_id
        metrics["test_flight_ids"] = test_ids
        metrics["test_flight_folders"] = test_folders
        split_summaries.append(metrics)
        rows_for_metrics.append({
            "split_id": split_id,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics.get("roc_auc"),
        })

    aggregate = aggregate_results(split_summaries)
    summary = {
        "task": "BASiC binary fault detection (Failure vs No Failure)",
        "n_flights": int(len(samples)),
        "class_distribution": {
            "NoFailure": int(np.sum(y == 0)),
            "Failure": int(np.sum(y == 1)),
        },
        "classes": {str(i): IDX2LABEL[i] for i in range(2)},
        "feature_profile": args.feature_profile,
        "stats_mode": args.stats_mode,
        "window_sec": args.window_sec,
        "model": args.model,
        "loss": args.loss if args.model == "mlp" else None,
        "test_size": args.test_size,
        "repeats": args.repeats,
        "epochs": args.epochs if args.model == "mlp" else None,
        "batch_size": args.batch_size if args.model == "mlp" else None,
        "lr": args.lr if args.model == "mlp" else None,
        "dropout": args.dropout if args.model == "mlp" else None,
        "hidden_dims": args.hidden_dims if args.model == "mlp" else None,
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
    cm_df = pd.DataFrame(mean_cm, index=[IDX2LABEL[i] for i in range(2)], columns=[IDX2LABEL[i] for i in range(2)])
    cm_df.to_csv(out_dir / "confusion_matrix_mean.csv", encoding="utf-8-sig")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(json.dumps(summary["metrics_mean"], ensure_ascii=False, indent=2))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary fault detection for BASiC Processed Data: Failure vs No Failure")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--metadata-csv", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/failure_vs_nofailure_mlp")
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--feature-profile", type=str, default="curated", choices=["curated", "full_numeric"])
    parser.add_argument("--stats-mode", type=str, default="enhanced", choices=["enhanced", "mean_std"])
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "rf", "xgb"])
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
    parser.add_argument("--normal-extra-modes", nargs="+", default=["q1", "q3"])
    parser.add_argument("--no-failure-anchor-mode", type=str, default="middle", choices=["middle", "q1", "q3"])
    args = parser.parse_args()
    run_experiment(args)
