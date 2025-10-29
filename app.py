import os
import json
import io
import re
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier  # noqa: F401 (for joblib compatibility if needed)
from scipy.signal import find_peaks, spectrogram
from scipy.stats import skew, kurtosis

import matplotlib.pyplot as plt
import streamlit as st


RAW_SIGNAL_LEN = 200
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
ARTIFACTS_PATH = os.path.join(MODEL_DIR, "artifacts.json")

# ----------------------------- Utilities -----------------------------
def read_artifacts() -> dict:
    if not os.path.exists(ARTIFACTS_PATH):
        st.error(f"artifacts.json not found at {ARTIFACTS_PATH}")
        st.stop()
    with open(ARTIFACTS_PATH, "r") as f:
        return json.load(f)


def _resolve_path(p: str) -> str:
    """Resolve artifact path to absolute path based on app directory."""
    if os.path.isabs(p):
        return p
    base_dir = os.path.dirname(__file__)
    return os.path.normpath(os.path.join(base_dir, p))


@st.cache_resource(show_spinner=False)
def load_models():
    artifacts = read_artifacts()

    scaler_path = _resolve_path(artifacts.get("scaler_tab", ""))
    rf_path = _resolve_path(artifacts.get("rf_full", ""))
    xgb_path = _resolve_path(artifacts.get("xgb_full", ""))
    meta_path = _resolve_path(artifacts.get("meta_lr", ""))

    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    rf_full = joblib.load(rf_path) if os.path.exists(rf_path) else None
    xgb_full = joblib.load(xgb_path) if os.path.exists(xgb_path) else None
    meta_lr = joblib.load(meta_path) if os.path.exists(meta_path) else None

    return {
        "scaler": scaler,
        "rf": rf_full,
        "xgb": xgb_full,
        "meta": meta_lr,
    }


def pad_or_trim(signal: np.ndarray, target_len: int = RAW_SIGNAL_LEN) -> np.ndarray:
    sig = np.asarray(signal, dtype=np.float32).flatten()
    if len(sig) >= target_len:
        return sig[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[: len(sig)] = sig
    return out


def normalize_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = float(x.mean())
    sd = float(x.std())
    if sd < 1e-8:
        sd = 1.0
    return (x - mu) / sd


def extract_features(beat_200: np.ndarray, rr_interval_sec: Optional[float] = None) -> Tuple[np.ndarray, dict]:
    b = np.asarray(beat_200, dtype=np.float32)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

    mean_v = float(np.mean(b))
    std_v = float(np.std(b))
    min_v = float(np.min(b))
    max_v = float(np.max(b))
    energy_v = float(np.sum(np.square(b)))
    skew_v = float(skew(b))
    kurt_v = float(kurtosis(b))

    peaks, _ = find_peaks(b, distance=30)
    r_peak_amp = float(b[peaks].max()) if len(peaks) > 0 else np.nan

    f, t, Sxx = spectrogram(b, fs=360, nperseg=32)
    spectral_energy = float(np.sum(Sxx))

    rr_v = float(rr_interval_sec) if rr_interval_sec is not None else 0.0

    feature_order = [
        "mean",
        "std",
        "min",
        "max",
        "energy",
        "skewness",
        "kurtosis",
        "r_peak_amplitude",
        "rr_interval_sec",
        "spectral_energy",
    ]

    feature_values = np.array(
        [mean_v, std_v, min_v, max_v, energy_v, skew_v, kurt_v, r_peak_amp, rr_v, spectral_energy],
        dtype=np.float32,
    )

    feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_values, {k: v for k, v in zip(feature_order, feature_values)}


def build_meta_features(models, tab_scaled: np.ndarray) -> Tuple[np.ndarray, dict]:
    rf = models["rf"]
    xgb = models["xgb"]

    rf_prob = rf.predict_proba(tab_scaled)[:, 1] if rf is not None else np.zeros((tab_scaled.shape[0],), dtype=np.float32)
    xgb_prob = xgb.predict_proba(tab_scaled)[:, 1] if xgb is not None else np.zeros((tab_scaled.shape[0],), dtype=np.float32)

    # pad a dummy LightGBM column (zero) to match meta-learner input
    lgb_dummy = np.zeros((tab_scaled.shape[0],), dtype=np.float32)

    meta_feats = np.column_stack([rf_prob, xgb_prob, lgb_dummy])
    # Only include actually loaded base models in base_probs (exclude dummy)
    base_probs = {}
    if models.get("rf") is not None:
        base_probs["rf"] = float(rf_prob[0])
    if models.get("xgb") is not None:
        base_probs["xgb"] = float(xgb_prob[0])
    return meta_feats, base_probs



def predict_probability(beat: np.ndarray, rr_interval: Optional[float], models, meta_weight: float) -> Tuple[float, float, float, dict, dict]:
    beat = pad_or_trim(beat, RAW_SIGNAL_LEN)
    features_vec, feature_dict = extract_features(beat, rr_interval)

    scaler = models["scaler"]
    if scaler is None:
        raise RuntimeError("Tabular scaler not found. Ensure saved_models/scaler_tab.pkl exists.")

    tab_scaled = scaler.transform(features_vec.reshape(1, -1))
    meta_feats, base_probs = build_meta_features(models, tab_scaled)

    meta = models["meta"]
    if meta is None:
        raise RuntimeError("Meta-learner not found. Ensure saved_models/meta_lr.pkl exists.")

    meta_prob = float(meta.predict_proba(meta_feats)[:, 1][0])

    # Compute mean over actual available base models only (no dummy)
    available = [v for v in base_probs.values() if not np.isnan(v)]
    base_mean = float(np.mean(available)) if available else meta_prob

    # Blend meta and base model predictions
    final_prob = float(meta_weight * meta_prob + (1.0 - meta_weight) * base_mean)
    return final_prob, meta_prob, base_mean, feature_dict, base_probs


def plot_ecg_with_highlights(beat: np.ndarray, threshold: float = 2.0):
    beat = pad_or_trim(beat, RAW_SIGNAL_LEN)
    z = normalize_zscore(beat)
    anomaly_mask = np.abs(z) > threshold

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(beat, lw=1.3)

    if anomaly_mask.any():
        idx = np.where(anomaly_mask)[0]
        splits = np.where(np.diff(idx) > 1)[0]
        segments = np.split(idx, splits + 1)
        for seg in segments:
            ax.axvspan(int(seg[0]), int(seg[-1]), color="red", alpha=0.18)

    ax.set_title("ECG (200 samples) with highlighted anomalies (|z| > %.1f)" % threshold)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)

# ----------------------------- UI -----------------------------
st.set_page_config(page_title="ECG Anomaly Detector", page_icon="🫀", layout="centered")
st.title("ECG Anomaly Detector (Meta-Stacked)")
st.caption("Upload an ECG segment (200 samples preferred). The app extracts features and predicts Abnormal probability.")

with st.sidebar:
    st.header("Models")
    st.write("Loading artifacts from:")
    st.code(MODEL_DIR)
    models = load_models()
    base_loaded = [name for name in ["rf", "xgb"] if models.get(name) is not None]
    ok = (models.get("scaler") is not None) and (models.get("meta") is not None) and (len(base_loaded) >= 1)
    if ok:
        st.success(f"Models loaded: scaler, meta, base={', '.join(base_loaded)}")
    else:
        st.warning("Missing some models. Need scaler + meta + at least one base (RF/XGB).")

    meta_weight = st.slider(
        "Meta weight in final blend",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Final = meta_weight * meta_prob + (1 - meta_weight) * mean(base_probs)"
    )

uploaded = st.file_uploader("Upload ECG file (.csv or .txt)", type=["csv", "txt"])

st.markdown("""
Accepted formats:
- CSV with a single column or row of numbers (200 values preferred).  
- TXT with whitespace/comma-separated numbers.  
Optionally, provide RR interval (sec) below.
""")

rr_val = st.number_input("Optional RR interval (seconds)", min_value=0.0, max_value=3.0, value=0.0, step=0.01)


def parse_numeric_payload(file_bytes: bytes) -> np.ndarray:
    text = file_bytes.decode("utf-8", errors="ignore")
    tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(tokens) >= 2:
        return np.array([float(t) for t in tokens], dtype=np.float32)
    try:
        df = pd.read_csv(io.StringIO(text), header=None)
        vals = pd.to_numeric(df.values.reshape(-1), errors='coerce').astype(np.float32)
        vals = vals[~np.isnan(vals)]
        return vals
    except Exception:
        return np.array([], dtype=np.float32)


if uploaded is not None:
    try:
        raw_vals = parse_numeric_payload(uploaded.read())
        if raw_vals.size == 0:
            st.warning("No numbers parsed from file.")
        else:
            st.subheader("Preview")
            st.write(f"Parsed {raw_vals.size} samples. Displaying first 200 after padding/trimming.")
            plot_ecg_with_highlights(raw_vals)

            with st.spinner("Predicting..."):
                prob, meta_prob, base_mean, feat_dict, base_probs = predict_probability(
                    raw_vals, rr_val if rr_val > 0 else None, models, meta_weight
                )

            st.subheader("Prediction")
            st.metric("p(Abnormal)", f"{prob:.3f}", help="Estimated probability of abnormal beat.")
            pred_label = "Abnormal" if prob >= 0.5 else "Normal"
            st.success(f"Predicted label: {pred_label}")

            # Explain decision
            st.subheader("How this decision was made")
            st.markdown(
                f"- Final probability blends meta and base models ({', '.join(base_loaded)}).  \n"
                f"- Blending: p = {meta_weight:.2f} × meta({meta_prob:.3f}) + {(1 - meta_weight):.2f} × base_mean({base_mean:.3f}).  \n"
                f"- Decision rule: predict Abnormal if p ≥ 0.5, else Normal.  \n"
                f"- For this sample: p = {prob:.3f} {'<' if prob < 0.5 else '≥'} 0.5 → {pred_label}."
            )

            st.caption("Base model probabilities (inputs to meta-learner / base_mean):")
            st.table(pd.DataFrame([base_probs]))

            st.subheader("Extracted Features")
            feat_df = pd.DataFrame([feat_dict]).fillna(0.0)
            st.dataframe(feat_df, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to process file: {e}")
else:
    st.info("Upload a CSV/TXT file with ECG samples to begin.")
