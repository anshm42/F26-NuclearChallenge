from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
import streamlit as st

from leak_detection.cli import (
    evaluate_binary_artifact,
    evaluate_type_artifact,
    load_artifact,
    predict_scenario,
)

BINARY_MODEL = Path("artifacts/leak_model.joblib")
TYPE_MODEL = Path("artifacts/leak_type_model.joblib")
TEST_DATA = Path("ML_Dataset/Testing")
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
        --ink: #0d0d0d;
        --panel: #171717;
        --line: #777777;
        --signal: #e65a3b;
        --signal-dark: #8f321f;
        --paper: #f5f5f2;
    }
    .stApp {
        background: var(--ink);
        color: var(--paper);
    }
    .stApp::before,
    .stApp::after {
        content: "";
        position: fixed;
        z-index: 0;
        pointer-events: none;
        width: 260px;
        height: 430px;
        background: repeating-linear-gradient(
            135deg,
            transparent 0 68px,
            var(--signal) 69px 112px,
            var(--signal-dark) 113px 136px,
            transparent 137px 190px
        );
        opacity: 0.92;
    }
    .stApp::before { top: -105px; right: -35px; }
    .stApp::after { bottom: -250px; left: -95px; }
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stSidebar"] { position: relative; z-index: 1; }
    h1, h2, h3, [data-testid="stMetricValue"] {
        font-family: "Avenir Next Condensed", "DIN Condensed", sans-serif;
        letter-spacing: 0.025em;
    }
    h1 {
        color: var(--paper);
        font-size: clamp(3rem, 7vw, 6.5rem);
        font-weight: 400;
        line-height: 0.95;
        text-transform: none;
    }
    h1::after {
        content: "";
        display: block;
        width: min(78vw, 920px);
        margin-top: 1.6rem;
        border-bottom: 1px solid #b8b8b8;
    }
    h2, h3 { color: var(--paper); }
    p, label, [data-testid="stCaptionContainer"] { color: #d9d9d6; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: #121212;
        border-right: 1px solid #3a3a3a;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(23, 23, 23, 0.96);
        border: 1px dashed var(--signal);
        border-radius: 0;
        min-height: 12rem;
    }
    [data-testid="stMetric"] {
        background: rgba(23, 23, 23, 0.96);
        border-top: 4px solid var(--signal);
        padding: 1rem;
    }
    [data-testid="stAlert"] {
        background: rgba(230, 90, 59, 0.1);
        border-color: var(--signal);
        border-radius: 0;
    }
    hr { border-color: #595959; }
    .status-card {
        border: 1px solid #595959;
        border-left: 7px solid var(--signal);
        background: rgba(23, 23, 23, 0.98);
        padding: 1rem 1.25rem;
        margin: 0.5rem 0 1.25rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .status-card.alert {
        background: #b42318;
        border-color: #ef6657;
        border-left-color: #ffb4aa;
        color: white;
        font-weight: 800;
    }
    .eyebrow {
        color: var(--signal);
        font-family: "Avenir Next Condensed", "DIN Condensed", sans-serif;
        font-size: 1rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .stButton > button {
        background: var(--signal);
        border: 1px solid var(--signal);
        border-radius: 0;
        color: white;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .stButton > button:hover {
        background: #f06a4a;
        border-color: #f06a4a;
        color: white;
    }
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
    st.caption(f"Test data: `{TEST_DATA}`")
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
        status = "SCRAM! Leak Detected!" if alert else "No leak alert"
        status_class = "status-card alert" if alert else "status-card"
        st.markdown(
            f'<div class="{status_class}">{status}</div>', unsafe_allow_html=True
        )

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
            st.bar_chart(type_frame, x="Leak type", y="Probability", color="#e65a3b")

        with st.expander("Prediction details"):
            visible_result = {
                key: value
                for key, value in result.items()
                if alert
                or key not in {"predicted_leak_type", "leak_type_probabilities"}
            }
            st.json(visible_result)

st.divider()
st.subheader("Full test-set evaluation")
st.caption("Evaluate every complete simulation in the testing split with the frozen models. Do not use these results to tune the models.")
test_data_missing = not TEST_DATA.is_dir()
if test_data_missing:
    st.error(f"Test directory not found: {TEST_DATA}")

if st.button(
    "Evaluate full test set",
    disabled=bool(missing) or test_data_missing,
    help="Runs both models across ML_Dataset/Testing.",
):
    try:
        with st.spinner("Evaluating all test simulations…"):
            binary_metrics, _ = evaluate_binary_artifact(
                TEST_DATA, trusted_artifact(str(BINARY_MODEL))
            )
            type_metrics, _ = evaluate_type_artifact(
                TEST_DATA, trusted_artifact(str(TYPE_MODEL))
            )
            st.session_state["full_test_results"] = (
                binary_metrics,
                type_metrics,
            )
    except (KeyError, TypeError, ValueError, OSError, pd.errors.ParserError) as error:
        st.error(f"Could not evaluate the test set: {error}")

if "full_test_results" in st.session_state:
    binary_metrics, type_metrics = st.session_state["full_test_results"]

    st.markdown("### Binary leak detection")
    binary_columns = st.columns(5)
    binary_columns[0].metric("Runs", int(binary_metrics["runs"]))
    binary_columns[1].metric("Recall", f'{float(binary_metrics["recall"]):.1%}')
    binary_columns[2].metric("Precision", f'{float(binary_metrics["precision"]):.1%}')
    binary_columns[3].metric("False negatives", int(binary_metrics["false_negatives"]))
    binary_columns[4].metric("False positives", int(binary_metrics["false_positives"]))

    binary_matrix = pd.DataFrame(
        [
            [binary_metrics["true_negatives"], binary_metrics["false_positives"]],
            [binary_metrics["false_negatives"], binary_metrics["true_positives"]],
        ],
        index=["Actual no leak", "Actual leak"],
        columns=["Predicted no leak", "Predicted leak"],
    )
    st.markdown("#### Binary confusion matrix")
    st.dataframe(binary_matrix, width="stretch")

    st.markdown("### Leak-type classification")
    type_columns = st.columns(4)
    type_columns[0].metric("Leak runs", int(type_metrics["runs"]))
    type_columns[1].metric("Accuracy", f'{float(type_metrics["accuracy"]):.1%}')
    type_columns[2].metric(
        "Top-2 accuracy", f'{float(type_metrics["top_2_accuracy"]):.1%}'
    )
    type_columns[3].metric("Log loss", f'{float(type_metrics["log_loss"]):.3f}')

    class_names = type_metrics["class_names"]
    type_matrix = pd.DataFrame(
        type_metrics["confusion_matrix"],
        index=[f"Actual {name}" for name in class_names],
        columns=[f"Predicted {name}" for name in class_names],
    )
    recall_frame = pd.DataFrame.from_dict(
        type_metrics["per_type_recall"], orient="index", columns=["Recall"]
    )
    recall_frame.index.name = "Leak type"
    st.markdown("#### Recall by leak type")
    st.dataframe(recall_frame.style.format("{:.1%}"), width="stretch")
    st.markdown("#### Leak-type confusion matrix")
    st.dataframe(type_matrix, width="stretch")
