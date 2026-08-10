import streamlit as st
import pandas as pd
import joblib


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

rf_model = joblib.load("gridguard_random_forest.pkl")
kmeans = joblib.load("gridguard_kmeans.pkl")
scaler = joblib.load("gridguard_scaler.pkl")
cluster_to_risk = joblib.load("cluster_to_risk.pkl")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GridGuard AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ==========================================================
   GENERAL PAGE
   ========================================================== */

.stApp {
    background-color: #FFFFFF;
    color: #1C2A38;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ==========================================================
   HEADER
   ========================================================== */

header[data-testid="stHeader"] {
    background-color: #FFFFFF;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background-color: #F4F6F8;
    border-right: 1px solid #D9E0E6;
}

section[data-testid="stSidebar"] h1 {
    color: #0A2540;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #1C2A38;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #536273;
}


/* ==========================================================
   HERO CARD
   ========================================================== */

.hero-card {
    background: linear-gradient(
        135deg,
        #F4F6F8,
        #EAF0F5
    );

    border: 1px solid #D6E0E8;
    border-left: 5px solid #0A2540;

    border-radius: 16px;

    padding: 35px;

    margin-bottom: 30px;

    box-shadow:
        0 4px 15px rgba(10, 37, 64, 0.06);
}


/* Main title */

.hero-title {
    font-size: 42px;
    font-weight: 800;

    color: #0A2540;

    letter-spacing: -1px;
}


/* GridGuard blue */

.hero-title span {
    color: #1E5A88;
}


/* Subtitle */

.hero-subtitle {
    color: #536273;

    font-size: 17px;

    margin-top: 5px;
}


/* Hero heading */

.hero-heading {
    color: #1C2A38;

    font-size: 24px;

    font-weight: 700;

    margin-top: 25px;
}


/* Hero description */

.hero-description {
    color: #536273;

    font-size: 15px;

    line-height: 1.7;

    margin-top: 10px;

    max-width: 900px;
}


/* ==========================================================
   SECTION HEADINGS
   ========================================================== */

.section-title {
    color: #0A2540;

    font-size: 22px;

    font-weight: 750;

    margin-top: 20px;

    margin-bottom: 15px;
}


/* ==========================================================
   INPUT CONTAINERS
   ========================================================== */

.input-card {
    background-color: #F4F6F8;

    border: 1px solid #D9E0E6;

    border-radius: 14px;

    padding: 20px;

    margin-bottom: 10px;
}


/* Input headings */

.input-heading {
    color: #0A2540;

    font-size: 18px;

    font-weight: 700;

    margin-bottom: 15px;
}


/* ==========================================================
   TEXT / NUMBER INPUTS
   ========================================================== */

div[data-baseweb="input"] {
    background-color: #FFFFFF !important;

    border: 1px solid #C8D2DC !important;

    border-radius: 8px !important;
}


/* Input text */

div[data-baseweb="input"] input {
    background-color: #FFFFFF !important;

    color: #1C2A38 !important;
}


/* Input focus */

div[data-baseweb="input"]:focus-within {
    border-color: #0A2540 !important;

    box-shadow:
        0 0 0 1px #0A2540 !important;
}


/* Number input buttons */

div[data-baseweb="input"] button {
    background-color: #F4F6F8 !important;

    color: #536273 !important;

    border: none !important;
}


/* Labels */

label {
    color: #1C2A38 !important;

    font-weight: 600 !important;
}


/* ==========================================================
   PRIMARY BUTTON
   ========================================================== */

.stButton > button {
    width: 100%;

    background-color: #0A2540 !important;

    color: #FFFFFF !important;

    border: none !important;

    border-radius: 9px !important;

    height: 48px;

    font-size: 16px;

    font-weight: 700;

    transition: 0.2s;
}


.stButton > button:hover {
    background-color: #153B5C !important;

    color: #FFFFFF !important;
}


/* ==========================================================
   RESULT CARDS
   ========================================================== */

.result-card {
    background-color: #F4F6F8;

    border: 1px solid #D9E0E6;

    border-radius: 14px;

    padding: 20px;

    text-align: center;
}


/* ==========================================================
   LOW RISK
   ========================================================== */

.risk-low {
    background-color: #E8F5EF;

    border: 2px solid #2E8B68;

    color: #176B4F;

    padding: 22px;

    border-radius: 14px;

    text-align: center;

    font-size: 30px;

    font-weight: 800;

    box-shadow:
        0 4px 12px rgba(46, 139, 104, 0.10);
}


/* ==========================================================
   MEDIUM RISK
   ========================================================== */

.risk-medium {
    background-color: #FFF6DF;

    border: 2px solid #D99A18;

    color: #946500;

    padding: 22px;

    border-radius: 14px;

    text-align: center;

    font-size: 30px;

    font-weight: 800;

    box-shadow:
        0 4px 12px rgba(217, 154, 24, 0.10);
}


/* ==========================================================
   HIGH RISK
   ========================================================== */

.risk-high {
    background-color: #FCEBEC;

    border: 2px solid #C94C4C;

    color: #9B2C2C;

    padding: 22px;

    border-radius: 14px;

    text-align: center;

    font-size: 30px;

    font-weight: 800;

    box-shadow:
        0 4px 12px rgba(201, 76, 76, 0.10);
}


/* ==========================================================
   METRIC CARDS
   ========================================================== */

div[data-testid="stMetric"] {
    background-color: #F4F6F8 !important;

    border: 1px solid #D9E0E6 !important;

    border-radius: 12px !important;

    padding: 18px !important;
}


div[data-testid="stMetric"] label {
    color: #536273 !important;
}


div[data-testid="stMetric"] div {
    color: #0A2540 !important;
}


/* ==========================================================
   DATAFRAME
   ========================================================== */

div[data-testid="stDataFrame"] {
    border: 1px solid #D9E0E6;

    border-radius: 10px;
}


/* ==========================================================
   INFO CARD
   ========================================================== */

.info-card {
    background-color: #F4F6F8;

    border: 1px solid #D9E0E6;

    border-radius: 14px;

    padding: 25px;

    margin-top: 20px;
}


.info-title {
    color: #0A2540;

    font-size: 20px;

    font-weight: 750;
}


.info-text {
    color: #536273;

    font-size: 14px;

    line-height: 1.7;

    margin-top: 10px;
}


/* ==========================================================
   DIVIDERS
   ========================================================== */

hr {
    border-color: #D9E0E6 !important;
}


/* ==========================================================
   SUCCESS / WARNING
   ========================================================== */

div[data-testid="stAlert"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h1 style="
            color:#0A2540;
            font-size:28px;
            margin-bottom:5px;
        ">
            ⚡ GridGuard AI
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style="
            color:#536273;
            font-size:14px;
            line-height:1.5;
        ">
            Regional Power Grid Supply Risk Assessment
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("System Information")

    st.write("**Clustering:** K-Means")
    st.write("**Classification:** Random Forest")
    st.write("**Clusters:** 3")
    st.write("**Risk Levels:** Low / Medium / High")

  

    st.caption(
        "GridGuard AI combines geographic K-Means "
        "clustering with Random Forest risk classification."
    )


# ============================================================
# HERO SECTION
# ============================================================

st.html(
    """
    <div class="hero-card">

        <div class="hero-title">
            Grid<span>Guard</span> AI
        </div>

        <div class="hero-subtitle">
            Regional Power Grid Supply Risk Prediction System
        </div>

        <div class="hero-heading">
            Evaluate Regional Grid Supply Conditions
        </div>

        <div class="hero-description">
            Enter the geographic and operational power-grid
            parameters to identify the regional supply risk
            using K-Means clustering and Random Forest
            classification.
        </div>

    </div>
    """
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    """
    <div class="section-title">
        Grid Parameters
    </div>
    """,
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


# ============================================================
# GEOGRAPHIC INFORMATION
# ============================================================

with col1:

    st.html(
        """
        <div class="input-card">

            <div class="input-heading">
                🌍 Geographic Information
            </div>

        </div>
        """
    )

    latitude = st.number_input(
        "Latitude",
        value=20.5,
        format="%.6f"
    )

    longitude = st.number_input(
        "Longitude",
        value=73.2,
        format="%.6f"
    )


# ============================================================
# DEMAND AND ENERGY
# ============================================================

with col2:

    st.html(
        """
        <div class="input-card">

            <div class="input-heading">
                ⚡ Demand & Energy
            </div>

        </div>
        """
    )

    max_demand = st.number_input(
        "Max Demand Met",
        value=1400.0
    )

    energy_met = st.number_input(
        "Energy Met",
        value=20.0
    )

    energy_shortage = st.number_input(
        "Energy Shortage",
        value=1.0
    )


# ============================================================
# GRID CONDITIONS
# ============================================================

with col3:

    st.html(
        """
        <div class="input-card">

            <div class="input-heading">
                🔌 Grid Conditions
            </div>

        </div>
        """
    )

    peak_shortage = st.number_input(
        "Shortage During Peak",
        value=15.0
    )

    drawl_schedule = st.number_input(
        "Drawl Schedule",
        value=35.0
    )

    od_ud = st.number_input(
        "OD(+) / UD(-)",
        value=2.0
    )

    max_od = st.number_input(
        "Max OD",
        value=30.0
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")

predict_button = st.button(
    "⚡ ANALYZE GRID RISK",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATA
    # --------------------------------------------------------

    new_data = pd.DataFrame(
        [{
            "latitude": latitude,
            "longitude": longitude,
            "Max Demand Met": max_demand,
            "Shortage During Peak": peak_shortage,
            "Energy Met": energy_met,
            "Drawl Schedule": drawl_schedule,
            "OD(+) / UD(-)": od_ud,
            "Max OD": max_od,
            "Energy Shortage": energy_shortage
        }]
    )


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf_prediction = rf_model.predict(
        new_data
    )[0]

    probabilities = rf_model.predict_proba(
        new_data
    )[0]


    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    scaled_data = scaler.transform(
        new_data
    )

    cluster = int(
        kmeans.predict(
            scaled_data
        )[0]
    )

    kmeans_risk = cluster_to_risk[
        cluster
    ]


    # ========================================================
    # RISK ASSESSMENT
    # ========================================================

    st.divider()

    st.markdown(
        """
        <div class="section-title">
            Risk Assessment
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # RISK RESULT
    # --------------------------------------------------------

    risk_class = rf_prediction.lower()

    st.html(
        f"""
        <div class="risk-{risk_class}">
            {rf_prediction.upper()} RISK
        </div>
        """
    )


    st.write("")


    # ========================================================
    # MODEL RESULTS
    # ========================================================

    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Random Forest Risk",
            rf_prediction
        )


    # with result_col2:

    #     st.metric(
    #         "K-Means Cluster",
    #         f"Cluster {cluster}"
    #     )


    with result_col3:

        st.metric(
            "K-Means Risk",
            kmeans_risk
        )


    st.write("")


    # ========================================================
    # MODEL AGREEMENT
    # ========================================================

    if rf_prediction == kmeans_risk:

        st.success(
            "✓ K-Means and Random Forest agree on the risk category."
        )

    else:

        st.warning(
            "⚠ K-Means and Random Forest produced different "
            "risk categories."
        )


    # ========================================================
    # RANDOM FOREST PROBABILITY
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
            Random Forest Risk Probability
        </div>
        """,
        unsafe_allow_html=True
    )


    probability_data = pd.DataFrame(
        {
            "Risk": rf_model.classes_,
            "Probability (%)": probabilities * 100
        }
    )


    probability_data[
        "Probability (%)"
    ] = probability_data[
        "Probability (%)"
    ].round(2)


    for _, row in probability_data.iterrows():

        st.write(
            f"**{row['Risk']} Risk — "
            f"{row['Probability (%)']}%**"
        )

        st.progress(
            int(row["Probability (%)"])
        )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.divider()

    st.markdown(
        """
        <div class="section-title">
            Input Summary
        </div>
        """,
        unsafe_allow_html=True
    )


    summary_data = pd.DataFrame(
        {
            "Parameter": [
                "Latitude",
                "Longitude",
                "Max Demand Met",
                "Shortage During Peak",
                "Energy Met",
                "Drawl Schedule",
                "OD(+) / UD(-)",
                "Max OD",
                "Energy Shortage"
            ],

            "Value": [
                latitude,
                longitude,
                max_demand,
                peak_shortage,
                energy_met,
                drawl_schedule,
                od_ud,
                max_od,
                energy_shortage
            ]
        }
    )


    st.dataframe(
        summary_data,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# ABOUT GRIDGUARD AI
# ============================================================

st.divider()

st.html(
    """
    <div class="info-card">

        <div class="info-title">
            About GridGuard AI
        </div>

        <div class="info-text">

            GridGuard AI evaluates regional power-grid supply
            conditions using a two-stage machine-learning approach.

            K-Means clustering identifies natural groups of
            grid operating conditions. The resulting clusters
            are analyzed and mapped to Low, Medium, and High
            risk categories.

            Random Forest then classifies the derived risk
            categories using geographic and operational
            power-grid features.

        </div>

    </div>
    """
)