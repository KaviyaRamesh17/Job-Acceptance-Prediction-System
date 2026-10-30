import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Job Acceptance Prediction",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Job Acceptance Prediction System")
st.subheader("HR Analytics - Data Driven Insights")

@st.cache_data
def load_data():
    return pd.read_csv("HR_Job_Placement_Cleaned.csv")

df = load_data()

# Load trained model
model = joblib.load("random_forest_model.pkl")
features = joblib.load("model_features.pkl")

# KPI calculations
total = len(df)
placed = (df["status"] == "Placed").sum()
not_placed = (df["status"] == "Not Placed").sum()
placement_rate = placed / total * 100

st.header("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Candidates", f"{total:,}")
col2.metric("Placed Candidates", f"{placed:,}")
col3.metric("Not Placed", f"{not_placed:,}")
col4.metric("Placement Rate", f"{placement_rate:.2f}%")

st.divider()

# Model performance
st.header("🤖 Machine Learning Model")

col1, col2 = st.columns(2)

col1.metric("Selected Model", "Random Forest")
col2.metric("Model Accuracy", "88.90%")

# Placement distribution
st.header("📈 Placement Analysis")

status_counts = df["status"].value_counts().reset_index()
status_counts.columns = ["Status", "Count"]

fig1 = px.pie(
    status_counts,
    names="Status",
    values="Count",
    title="Candidate Placement Distribution"
)

st.plotly_chart(fig1, use_container_width=True)

# Experience analysis
fig2 = px.histogram(
    df,
    x="experience_category",
    color="status",
    barmode="group",
    title="Experience Category vs Placement"
)

st.plotly_chart(fig2, use_container_width=True)

# Skills analysis
fig3 = px.histogram(
    df,
    x="skills_match_level",
    color="status",
    barmode="group",
    title="Skills Match Level vs Placement"
)

st.plotly_chart(fig3, use_container_width=True)

# Feature importance
st.header("⭐ Top 10 Important Features")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(10)

fig4 = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Random Forest Feature Importance"
)

st.plotly_chart(fig4, use_container_width=True)

st.success("Dashboard Loaded Successfully!")

st.caption(
    "Note: Placed and Not Placed represent dataset placement outcomes "
    "used as a proxy for job acceptance."
)
# Candidate Prediction Section

st.divider()
st.header("🔍 Candidate Placement Prediction")

st.write("Enter candidate details to predict the placement outcome.")

with st.form("candidate_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        age_years = st.number_input("Age", 21, 34, 27)
        gender = st.selectbox("Gender", ["Female", "Male"])
        ssc_percentage = st.number_input("SSC Percentage", 50.0, 95.0, 70.0)
        hsc_percentage = st.number_input("HSC Percentage", 50.0, 95.0, 72.0)
        degree_percentage = st.number_input("Degree Percentage", 50.0, 95.0, 74.0)
        degree_specialization = st.selectbox(
            "Degree Specialization",
            ["Others", "Electronics", "IT", "CS", "Mechanical"]
        )
        technical_score = st.number_input("Technical Score", 40.0, 100.0, 68.0)
        aptitude_score = st.number_input("Aptitude Score", 40.0, 100.0, 64.0)
        communication_score = st.number_input("Communication Score", 40.0, 100.0, 66.0)

    with col2:
        skills_match_percentage = st.number_input("Skills Match Percentage", 45.0, 100.0, 74.0)
        certifications_count = st.number_input("Certifications Count", 0, 8, 1)
        internship_experience = st.selectbox("Internship", ["Yes", "No"])
        years_of_experience = st.number_input("Years of Experience", 0.0, 5.0, 1.0)
        career_switch_willingness = st.selectbox(
            "Career Switch",
            ["Willing", "Not Willing", "Unknown"]
        )
        relevant_experience = st.selectbox(
            "Relevant Experience",
            ["Relevant", "Not Relevant", "Unknown"]
        )
        previous_ctc_lpa = st.number_input("Previous CTC (LPA)", 1.0, 14.0, 4.8)
        expected_ctc_lpa = st.number_input("Expected CTC (LPA)", 3.0, 27.0, 8.0)

    with col3:
        company_tier = st.selectbox("Company Tier", ["Tier1", "Tier2", "Tier3"])
        job_role_match = st.selectbox(
            "Job Role Match",
            ["Matched", "Not Matched", "Unknown"]
        )
        competition_level = st.selectbox(
            "Competition Level",
            ["Low", "Medium", "High"]
        )
        bond_requirement = st.selectbox(
            "Bond Requirement",
            ["Required", "Not Required"]
        )
        notice_period_days = st.number_input("Notice Period (Days)", 0.0, 90.0, 20.0)
        layoff_history = st.selectbox("Layoff History", ["Yes", "No", "Unknown"])
        employment_gap_months = st.number_input("Employment Gap (Months)", 0.0, 18.0, 3.0)
        relocation_willingness = st.selectbox(
            "Relocation Willingness",
            ["Willing", "Not Willing", "Unknown"]
        )

    predict_button = st.form_submit_button("Predict Placement")

if predict_button:

    candidate = pd.DataFrame([{
        "age_years": age_years,
        "gender": gender,
        "ssc_percentage": ssc_percentage,
        "hsc_percentage": hsc_percentage,
        "degree_percentage": degree_percentage,
        "degree_specialization": degree_specialization,
        "technical_score": technical_score,
        "aptitude_score": aptitude_score,
        "communication_score": communication_score,
        "skills_match_percentage": skills_match_percentage,
        "certifications_count": certifications_count,
        "internship_experience": internship_experience,
        "years_of_experience": years_of_experience,
        "career_switch_willingness": career_switch_willingness,
        "relevant_experience": relevant_experience,
        "previous_ctc_lpa": previous_ctc_lpa,
        "expected_ctc_lpa": expected_ctc_lpa,
        "company_tier": company_tier,
        "job_role_match": job_role_match,
        "competition_level": competition_level,
        "bond_requirement": bond_requirement,
        "notice_period_days": notice_period_days,
        "layoff_history": layoff_history,
        "employment_gap_months": employment_gap_months,
        "relocation_willingness": relocation_willingness
    }])

    candidate["experience_category"] = pd.cut(
        candidate["years_of_experience"],
        bins=[-1, 0, 2, float("inf")],
        labels=["Fresher", "Junior", "Senior"]
    )

    candidate["academic_performance"] = pd.cut(
        candidate["degree_percentage"],
        bins=[0, 60, 75, 100],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    candidate["skills_match_level"] = pd.cut(
        candidate["skills_match_percentage"],
        bins=[0, 60, 75, 100],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    candidate["evaluation_score"] = candidate[
        ["technical_score", "aptitude_score", "communication_score"]
    ].mean(axis=1)

    candidate["evaluation_performance"] = pd.cut(
        candidate["evaluation_score"],
        bins=[0, 60, 75, 100],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    candidate_encoded = pd.get_dummies(
        candidate,
        drop_first=True,
        dtype=int
    )

    candidate_encoded = candidate_encoded.reindex(
        columns=features,
        fill_value=0
    )

    prediction = model.predict(candidate_encoded)[0]
    probability = model.predict_proba(candidate_encoded)[0][1] * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Predicted Result: Placed")
    else:
        st.warning("Predicted Result: Not Placed")

    st.metric("Model Estimated Placement Probability", f"{probability:.2f}%")