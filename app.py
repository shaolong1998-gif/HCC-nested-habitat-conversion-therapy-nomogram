import math
from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="Nested Habitat Model for Conversion Therapy Efficacy in HCC",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f5f8fc; }
    .block-container { max-width: 1560px; padding-top: 0.9rem; padding-bottom: 1.2rem; }
    h1, h2, h3 { color: #274c77; }
    h1 { font-size: 1.75rem; line-height: 1.25; margin-bottom: 0.15rem; }
    h2, h3 { margin-bottom: 0.35rem; }
    div[data-testid="stMetric"] {
        background: #fcf7f3;
        border-left: 5px solid #d17b4d;
        border-radius: 7px;
        padding: 10px 14px;
    }
    div[data-testid="stAlert"] { padding: 0.55rem 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Coefficients from the locked R glm fitted on the training cohort (n = 280).
COEFFICIENTS = {
    "intercept": -2.1230442776584693,
    "number_of_tumors": -0.5409869632696711,
    "delta_plt": -1.566511364161311,
    "p_afp": -0.05232433373709918,
    "hepatitis_b": -0.8170661864218507,
    "nested_habitat_score": 9.987472850166286,
}


def logistic_probability(linear_predictor: float) -> float:
    """Calculate an overflow-safe inverse-logit probability."""
    if linear_predictor >= 0:
        return 1 / (1 + math.exp(-linear_predictor))
    exp_value = math.exp(linear_predictor)
    return exp_value / (1 + exp_value)


st.title(
    "Nested Habitat Model for Evaluating the Efficacy of Conversion Therapy "
    "in Hepatocellular Carcinoma"
)
st.write(
    "This interactive calculator uses the locked logistic regression model "
    "developed in the training cohort. All calculations occur within the "
    "application; no patient-level data are stored."
)

calculator_column, nomogram_column = st.columns([0.42, 0.58], gap="large")

with calculator_column:
    st.subheader("Patient Variables")
    input_left, input_right = st.columns(2, gap="small")
    with input_left:
        number_of_tumors = st.selectbox(
            "Number of Tumors",
            options=[1, 2, 3],
            index=0,
            help="Select the number of tumors as encoded in the development dataset.",
        )
        delta_plt = st.number_input(
            "DELTA%-PLT",
            value=0.0,
            step=0.001,
            format="%.3f",
            help="(P-Platelets - Platelets) / Platelets",
        )
        nested_habitat_score = st.number_input(
            "Nested Habitat Score",
            min_value=0.0,
            max_value=1.0,
            value=0.200,
            step=0.001,
            format="%.3f",
            help="texture_low_low_svm_score",
        )
    with input_right:
        hepatitis_b = st.selectbox(
            "Hepatitis B",
            options=[0, 1],
            index=1,
            format_func=lambda value: "Yes (1)" if value == 1 else "No (0)",
        )
        p_afp = st.number_input(
            "P-AFP",
            min_value=0.0,
            value=5.0,
            step=0.001,
            format="%.3f",
        )

    linear_predictor = (
        COEFFICIENTS["intercept"]
        + COEFFICIENTS["number_of_tumors"] * number_of_tumors
        + COEFFICIENTS["delta_plt"] * delta_plt
        + COEFFICIENTS["p_afp"] * p_afp
        + COEFFICIENTS["hepatitis_b"] * hepatitis_b
        + COEFFICIENTS["nested_habitat_score"] * nested_habitat_score
    )
    probability = logistic_probability(linear_predictor)

    st.subheader("Estimated Outcome")
    st.metric(
        "Predicted Probability of Conversion Therapy Response",
        f"{probability:.1%}",
    )
    st.caption(f"Linear predictor: {linear_predictor:.3f}")
    st.info(
        "Locked model performance: training cohort (n = 280), AUC = 0.815; "
        "test cohort (n = 70), AUC = 0.730."
    )
    st.warning(
        "For research and manuscript presentation only. This tool must not be "
        "used as a stand-alone clinical decision tool."
    )

with nomogram_column:
    st.subheader("Nomogram")
    nomogram_path = Path(__file__).with_name(
        "hcc-nested-habitat-conversion-therapy-nomogram.png"
    )
    if nomogram_path.exists():
        st.image(
            str(nomogram_path),
            caption=(
                "Nested Habitat Model for Evaluating the Efficacy of Conversion "
                "Therapy in Hepatocellular Carcinoma"
            ),
        use_column_width=True,
        )
    else:
        st.error("The nomogram image file is missing from the application folder.")
