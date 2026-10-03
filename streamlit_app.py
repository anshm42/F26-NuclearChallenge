from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
import streamlit as st

from leak_detection.cli import load_artifact, predict_scenario

BINARY_MODEL = Path("artifacts/leak_model.joblib")
TYPE_MODEL = Path("artifacts/leak_type_model.joblib")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

st.set_page_config(
    page_title="Leak Detection Model",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #071417;
        --panel: #102327;
        --line: #315057;
        --signal: #f2b544;
        --cool: #76d7cf;
        --paper: #e7eee9;
    }
    .stApp {
        background:
            linear-gradient(rgba(118, 215, 207, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(118, 215, 207, 0.035) 1px, transparent 1px),
            var(--ink);
        background-size: 32px 32px;
        color: var(--paper);
    }
    h1, h2, h3, [data-testid="stMetricValue"] {
        font-family: "Avenir Next Condensed", "Helvetica Neue", sans-serif;
        letter-spacing: 0.035em;
    }
    h1 { text-transform: uppercase; }
    [data-testid="stSidebar"] { background: #091a1e; border-right: 1px solid var(--line); }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(16, 35, 39, 0.92);
        border: 1px dashed var(--cool);
        border-radius: 2px;
        min-height: 12rem;
    }
    [data-testid="stMetric"] {
        background: rgba(16, 35, 39, 0.92);
        border-top: 3px solid var(--cool);
        padding: 1rem;
    }
    .status-card {
        border: 1px solid var(--line);
        border-left: 6px solid var(--signal);
        background: rgba(16, 35, 39, 0.96);
        padding: 1rem 1.25rem;
        margin: 0.5rem 0 1.25rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .eyebrow { color: var(--cool); font-size: 0.78rem; letter-spacing: 0.18em; text-transform: uppercase; }
    .stButton > button { border-radius: 2px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def trusted_artifact(path: str) -> dict[str, Any]:
    return load_artifact(path)


st.markdown('<p class="eyebrow">NPPAD / Early-warning classifier</p>', unsafe_allow_html=True)
st.title("Leak Detection Model")
st.caption("Calibrated XGBoost analysis of one simulated reactor scenario.")
st.warning(
    "Research prototype using simulated NPPAD data. Not validated for plant operation or safety decisions.",
    icon="⚠️",
)

with st.sidebar:
    st.subheader("System status")
    missing = [path for path in (BINARY_MODEL, TYPE_MODEL) if not path.is_file()]
    if missing:
        st.error("Missing model artifacts:\n" + "\n".join(str(path) for path in missing))
    else:
        st.success("Models ready")
    st.caption(f"Binary model: `{BINARY_MODEL}`")
    st.caption(f"Type model: `{TYPE_MODEL}`")
    st.divider()
    st.caption("Accepted input: one operation CSV, maximum 25 MB.")

uploaded = st.file_uploader(
    "Drop one scenario CSV",
    type=["csv"],
    help="The CSV must include the TIME column and compatible NPPAD sensor columns.",
)

if uploaded is not None:
    st.caption(f"Selected: `{uploaded.name}` · {uploaded.size / 1024:.1f} KB")

if st.button("Analyze scenario", type="primary", disabled=uploaded is None or bool(missing)):
    if uploaded is None:
        st.stop()
    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("CSV exceeds the 25 MB upload limit.")
        st.stop()

    try:
        with st.spinner("Reading the first model window…"):
            with TemporaryDirectory() as directory:
                csv_path = Path(directory) / Path(uploaded.name).name
                csv_path.write_bytes(uploaded.getvalue())
                result = predict_scenario(
                    csv_path,
                    trusted_artifact(str(BINARY_MODEL)),
                    trusted_artifact(str(TYPE_MODEL)),
                    input_name=uploaded.name,
                )
    except (KeyError, TypeError, ValueError, OSError, pd.errors.ParserError) as error:
        st.error(f"Could not analyze this CSV: {error}")
    else:
        probability = float(result["leak_probability"])
        threshold = float(result["alert_threshold"])
        alert = bool(result["leak_alert"])
        status = "Leak alert" if alert else "No leak alert"
        st.markdown(f'<div class="status-card">{status}</div>', unsafe_allow_html=True)

        metric_columns = st.columns(3 if alert else 2)
        metric_columns[0].metric("Leak probability", f"{probability:.1%}")
        metric_columns[1].metric("Alert threshold", f"{threshold:.1%}")

        if alert:
            metric_columns[2].metric(
                "Most likely type", str(result["predicted_leak_type"])
            )
            st.subheader("Conditional leak-type probabilities")
            st.caption(
                "Type estimates are secondary. The binary leak alert remains the primary decision."
            )
            type_frame = pd.DataFrame(
                {
                    "Leak type": list(result["leak_type_probabilities"]),
                    "Probability": list(result["leak_type_probabilities"].values()),
                }
            ).sort_values("Probability", ascending=False)
            st.bar_chart(type_frame, x="Leak type", y="Probability", color="#76d7cf")

        with st.expander("Prediction details"):
            visible_result = {
                key: value
                for key, value in result.items()
                if alert
                or key not in {"predicted_leak_type", "leak_type_probabilities"}
            }
            st.json(visible_result)
