import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import sys
import os
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Cyber Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# PROJECT PATH
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

# ============================================================
# SOURCE IMPORTS
# ============================================================
sys.path.append(str(PROJECT_ROOT))

from src.recommendations import get_recommendations
from src.severity import calculate_severity
from src.mitre_mapping import get_mitre_mapping

CONFIDENCE_THRESHOLD = 0.90

# ============================================================
# PROFESSIONAL LIGHT THEME
# ============================================================
st.markdown(
    """
    <style>
        .stApp {
            background: #f6f8fb;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: #172033;
        }

        p, label, div {
            font-family: "Segoe UI", Arial, sans-serif;
        }

        .top-header {
            background: #ffffff;
            border: 1px solid #e5e9f0;
            border-radius: 18px;
            padding: 24px 28px;
            margin-bottom: 18px;
            box-shadow: 0 4px 18px rgba(30, 41, 59, 0.05);
        }

        .top-title {
            font-size: 30px;
            font-weight: 700;
            color: #172033;
            margin-bottom: 5px;
        }

        .top-subtitle {
            font-size: 15px;
            color: #667085;
        }

        .online-badge {
            display: inline-block;
            background: #ecfdf3;
            color: #027a48;
            border: 1px solid #abefc6;
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 13px;
            font-weight: 600;
        }

        .section-title {
            color: #344054;
            font-size: 18px;
            font-weight: 700;
            margin: 8px 0 12px 0;
        }

        .status-card {
            background: #ffffff;
            border: 1px solid #e5e9f0;
            border-radius: 14px;
            padding: 16px 18px;
            min-height: 92px;
            box-shadow: 0 3px 12px rgba(30, 41, 59, 0.04);
        }

        .status-label {
            color: #667085;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .status-value {
            color: #172033;
            font-size: 17px;
            font-weight: 700;
            margin-top: 6px;
        }

        .status-dot {
            color: #12b76a;
        }

        .card {
            background: #ffffff;
            border: 1px solid #e5e9f0;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 16px rgba(30, 41, 59, 0.04);
        }

        .verdict-label {
            color: #667085;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .verdict-main {
            font-size: 26px;
            font-weight: 750;
            margin-top: 8px;
            color: #172033;
        }

        .verdict-threat {
            color: #b42318;
        }

        .verdict-suspicious {
            color: #b54708;
        }

        .verdict-benign {
            color: #027a48;
        }

        .severity-pill {
            display: inline-block;
            margin-top: 10px;
            padding: 7px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            background: #f2f4f7;
            color: #344054;
            border: 1px solid #e4e7ec;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid #e5e9f0;
            border-radius: 14px;
            padding: 15px 17px;
            box-shadow: 0 3px 12px rgba(30, 41, 59, 0.04);
        }

        .metric-label {
            color: #667085;
            font-size: 12px;
            font-weight: 600;
        }

        .metric-value {
            color: #172033;
            font-size: 22px;
            font-weight: 750;
            margin-top: 5px;
        }

        .model-card {
            background: #fbfcfe;
            border: 1px solid #e5e9f0;
            border-radius: 14px;
            padding: 18px;
            min-height: 145px;
        }

        .model-name {
            color: #344054;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .model-result {
            color: #172033;
            font-size: 21px;
            font-weight: 750;
            margin-top: 10px;
        }

        .model-detail {
            color: #667085;
            font-size: 13px;
            margin-top: 8px;
        }

        .rec-card {
            background: #fbfcfe;
            border: 1px solid #e5e9f0;
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }

        .rec-number {
            color: #175cd3;
            font-weight: 800;
            margin-right: 8px;
        }

        .rec-text {
            color: #344054;
            font-size: 14px;
            line-height: 1.5;
        }

        .mitre-id {
            font-size: 24px;
            color: #175cd3;
            font-weight: 800;
        }

        .mitre-technique {
            font-size: 16px;
            color: #172033;
            font-weight: 700;
            margin-top: 5px;
        }

        .mitre-tactic {
            color: #667085;
            font-size: 13px;
            margin-top: 5px;
        }

        .small-muted {
            color: #667085;
            font-size: 13px;
        }

        .stButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: 700;
            min-height: 44px;
        }

        [data-testid="stFileUploader"] {
            background: #ffffff;
            border: 1px dashed #cfd6e4;
            border-radius: 14px;
            padding: 8px;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e9f0;
            border-radius: 12px;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    model = joblib.load(MODELS_DIR / "xgboost_model.pkl")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    anomaly_model = joblib.load(MODELS_DIR / "isolation_forest.pkl")
    return model, label_encoder, anomaly_model


try:
    model, label_encoder, anomaly_model = load_models()
except Exception as e:
    st.error(f"Unable to load the trained models: {e}")
    st.stop()

# ============================================================
# HELPERS
# ============================================================
def prepare_features(row, model):
    actual_label = row["Label"] if "Label" in row.index else "Unknown"

    X = row.drop(labels=["Label"], errors="ignore").to_frame().T.copy()

    if "Protocol" in X.columns and not pd.api.types.is_numeric_dtype(X["Protocol"]):
        protocol_map = {
            "tcp": 6,
            "udp": 17,
            "icmp": 1,
            "ipv4-icmp": 1,
            "icmpv4": 1,
            "icmpv6": 58,
        }

        protocol_text = (
            X["Protocol"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        X["Protocol"] = protocol_text.map(protocol_map)

    model_features = getattr(model, "feature_names_in_", None)

    if model_features is not None:
        model_features = list(model_features)
        missing = [c for c in model_features if c not in X.columns]

        if missing:
            raise ValueError(
                "The uploaded dataset is missing model features: "
                + ", ".join(missing)
            )

        X = X[model_features]

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, actual_label


def decode_prediction(raw_prediction, encoder):
    try:
        encoded = int(raw_prediction)
        return str(encoder.inverse_transform([encoded])[0])
    except (ValueError, TypeError, IndexError):
        return str(raw_prediction)


def hybrid_decision(predicted_label, confidence, anomaly_status):
    if (
        predicted_label != "Benign"
        and confidence >= CONFIDENCE_THRESHOLD
    ):
        return f"Known Threat: {predicted_label}", "known"

    if (
        predicted_label == "Benign"
        and anomaly_status == "Normal"
        and confidence >= CONFIDENCE_THRESHOLD
    ):
        return "Benign", "benign"

    if (
        anomaly_status == "Anomaly"
        and (
            predicted_label == "Benign"
            or confidence < CONFIDENCE_THRESHOLD
        )
    ):
        return "Suspicious / Potential Novel Threat", "suspicious"

    return "Suspicious / Potential Novel Threat", "suspicious"


def get_shap_values(model, X, prediction):
    try:
        explainer = shap.TreeExplainer(model)
        result = explainer(X)
        values = result.values

        if isinstance(values, list):
            class_index = int(prediction)
            shap_values = np.asarray(values[class_index])[0]
        else:
            values = np.asarray(values)

            if values.ndim == 3:
                class_index = int(prediction)
                shap_values = values[0, :, class_index]
            elif values.ndim == 2:
                shap_values = values[0]
            else:
                shap_values = values.flatten()

        return np.asarray(shap_values)

    except Exception:
        return None


def verdict_class(decision_type):
    if decision_type == "known":
        return "verdict-threat"
    if decision_type == "suspicious":
        return "verdict-suspicious"
    return "verdict-benign"


# ============================================================
# HEADER
# ============================================================
header_left, header_right = st.columns([5, 1])

with header_left:
    st.markdown(
        """
        <div class="top-header">
            <div class="top-title">🛡️ AI Cyber Threat Detection System</div>
            <div class="top-subtitle">
                Network Security Intelligence Platform · Hybrid ML · Anomaly Detection · Explainable AI
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        """
        <div style="text-align:right; padding-top:22px;">
            <span class="online-badge">● SYSTEM ONLINE</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# MODEL STATUS
# ============================================================
st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)

with s1:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-label">Supervised Detection</div>
            <div class="status-value"><span class="status-dot">●</span> XGBoost Ready</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with s2:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-label">Unsupervised Detection</div>
            <div class="status-value"><span class="status-dot">●</span> Isolation Forest Ready</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with s3:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-label">Explainability</div>
            <div class="status-value"><span class="status-dot">●</span> SHAP Ready</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# ============================================================
# DATASET INPUT
# ============================================================
st.markdown('<div class="section-title">Dataset & Traffic Analysis</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a network traffic dataset",
    type=["parquet"],
    label_visibility="visible"
)

if uploaded_file is None:
    st.info("Upload a Parquet traffic dataset to start the analysis.")
    st.stop()

try:
    df = pd.read_parquet(uploaded_file)
except Exception as e:
    st.error(f"Unable to read the dataset: {e}")
    st.stop()

if "Label" not in df.columns:
    st.error("The uploaded dataset must contain a 'Label' column.")
    st.stop()

if len(df) == 0:
    st.error("The uploaded dataset contains no records.")
    st.stop()

d1, d2 = st.columns([4, 1])

with d1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Loaded Dataset</div>
            <div class="metric-value">{uploaded_file.name}</div>
            <div class="small-muted">{len(df):,} traffic records available for analysis</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with d2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Records</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="small-muted">Available</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

r1, r2 = st.columns([4, 1])

with r1:
    record_number = st.number_input(
        "Traffic Record Number (1 = first row)",
        min_value=1,
        max_value=len(df),
        value=1,
        step=1
    )
    record_number = int(record_number)

with r2:
    st.write("")
    st.write("")
    analyze_clicked = st.button("🔍 Analyze Traffic", type="primary", use_container_width=True)

selected_row = df.iloc[record_number - 1]

with st.expander("View Raw Traffic Features"):
    st.dataframe(
        selected_row.to_frame().T,
        use_container_width=True,
        hide_index=True
    )

if not analyze_clicked:
    st.info("Select a traffic record and click **Analyze Traffic**.")
    st.stop()

# ============================================================
# PREPARE SELECTED RECORD
# ============================================================
try:
    X, actual_label = prepare_features(selected_row, model)
except Exception as e:
    st.error(f"Feature preparation failed: {e}")
    st.stop()

try:
    # ========================================================
    # XGBOOST
    # ========================================================
    raw_prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    predicted_label = decode_prediction(
        raw_prediction,
        label_encoder
    )

    confidence = float(np.max(probabilities))

    # ========================================================
    # ISOLATION FOREST
    # ========================================================
    anomaly_prediction = anomaly_model.predict(X)[0]
    anomaly_score = float(
        anomaly_model.decision_function(X)[0]
    )

    anomaly_status = (
        "Anomaly"
        if anomaly_prediction == -1
        else "Normal"
    )

    # ========================================================
    # THREAT SEVERITY
    # ========================================================
    severity = calculate_severity(
        predicted_label,
        confidence,
        anomaly_status
    )

    # ========================================================
    # HYBRID DECISION
    # ========================================================
    final_decision, decision_type = hybrid_decision(
        predicted_label,
        confidence,
        anomaly_status
    )

    # ========================================================
    # MITRE ATT&CK
    # ========================================================
    mitre_info = get_mitre_mapping(predicted_label)

    # ========================================================
    # SECURITY RECOMMENDATIONS
    # ========================================================
    recommendations = get_recommendations(
        predicted_label,
        final_decision
    )[:3]

    # ========================================================
    # THREAT VERDICT + MODEL CONSENSUS
    # ========================================================
    st.write("")
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown(
            f"""
            <div class="card">
                <div class="verdict-label">Threat Verdict · Record {record_number}</div>
                <div class="verdict-main {verdict_class(decision_type)}">
                    {final_decision}
                </div>
                <div class="severity-pill">Severity: {severity}</div>
                <div style="margin-top:14px;" class="small-muted">
                    Actual dataset label: <b>{actual_label}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            """
            <div class="card">
                <div class="verdict-label">Model Consensus</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        mc1, mc2 = st.columns(2)

        with mc1:
            st.markdown(
                f"""
                <div class="model-card">
                    <div class="model-name">XGBoost</div>
                    <div class="model-result">{predicted_label}</div>
                    <div class="model-detail">Confidence · {confidence:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with mc2:
            st.markdown(
                f"""
                <div class="model-card">
                    <div class="model-name">Isolation Forest</div>
                    <div class="model-result">{anomaly_status}</div>
                    <div class="model-detail">Score · {anomaly_score:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ========================================================
    # SECURITY INTELLIGENCE METRICS
    # ========================================================
    st.write("")
    st.markdown('<div class="section-title">Security Intelligence</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    metrics = [
        ("Threat Severity", severity),
        ("Model Confidence", f"{confidence:.2%}"),
        ("Anomaly Score", f"{anomaly_score:.4f}"),
        ("Actual Label", str(actual_label)),
    ]

    for col, (label, value) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ========================================================
    # SHAP EXPLANATION
    # ========================================================
    st.write("")
    st.markdown('<div class="section-title">🔎 Explainable AI — SHAP</div>', unsafe_allow_html=True)
    st.caption("Why did the model make this prediction for the selected traffic record?")

    shap_values = get_shap_values(
        model,
        X,
        raw_prediction
    )

    if shap_values is None:
        st.warning("SHAP explanation could not be generated.")
    elif len(shap_values) != len(X.columns):
        st.warning("SHAP output size does not match the model features.")
    else:
        explanation_df = pd.DataFrame({
            "Feature": list(X.columns),
            "Value": X.iloc[0].values,
            "SHAP": shap_values
        })

        explanation_df["Abs_SHAP"] = explanation_df["SHAP"].abs()

        top_features = (
            explanation_df
            .sort_values("Abs_SHAP", ascending=False)
            .head(5)
            .copy()
        )

        top_features["Effect"] = np.where(
            top_features["SHAP"] >= 0,
            "Supporting prediction",
            "Opposing prediction"
        )

        top_features["SHAP"] = top_features["SHAP"].round(4)

        shap_display = top_features[
            ["Feature", "Value", "SHAP", "Effect"]
        ].copy()

        shap_display["Value"] = shap_display["Value"].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.number)) else str(x)
        )

        st.dataframe(
            shap_display,
            use_container_width=True,
            hide_index=True
        )

        feature_text = ", ".join(
            top_features["Feature"].head(3).tolist()
        )

        st.info(
            f"The prediction **{predicted_label}** was mainly influenced by: {feature_text}."
        )

    # ========================================================
    # MITRE + RECOMMENDATIONS
    # ========================================================
    st.write("")
    intel_left, intel_right = st.columns([1, 1.45])

    with intel_left:
        st.markdown('<div class="section-title">MITRE ATT&CK Mapping</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="card">
                <div class="mitre-id">{mitre_info['mitre_id']}</div>
                <div class="mitre-technique">{mitre_info['technique']}</div>
                <div class="mitre-tactic">Tactic · {mitre_info['tactic']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with intel_right:
        st.markdown('<div class="section-title">🛡️ Security Recommendations</div>', unsafe_allow_html=True)

        for i, recommendation in enumerate(recommendations, start=1):
            st.markdown(
                f"""
                <div class="rec-card">
                    <span class="rec-number">{i:02d}</span>
                    <span class="rec-text">{recommendation}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================
    st.write("")
    with st.expander("Technical Analysis Details"):
        st.write(f"**Final Decision:** {final_decision}")
        st.write(f"**Decision Type:** {decision_type}")
        st.write(f"**Actual Label:** {actual_label}")
        st.write(f"**Raw XGBoost Prediction:** `{raw_prediction}`")
        st.write(
            f"**XGBoost Probability Vector:** `{np.round(probabilities, 6).tolist()}`"
        )
        st.write(f"**Model Input Shape:** `{X.shape}`")
        st.write(f"**Number of Model Features:** `{len(X.columns)}`")
        st.write(
            "**Protocol value sent to model:** "
            f"`{X['Protocol'].iloc[0] if 'Protocol' in X.columns else 'N/A'}`"
        )

except Exception as e:
    st.error(
        "Prediction failed. Check the model files and the uploaded dataset features."
    )
    st.exception(e)
