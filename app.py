import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="Student Performance Analytics",
    layout="wide"
)

st.title("🎓 Student Performance Pattern Clustering")
st.write("Industry-grade ML system for academic intervention")

# =====================================================
# TRAIN MODEL (CACHED)
# =====================================================
@st.cache_resource
def train_model(df):

    df = df.copy()

    # ----- Attendance -----
    df["attendance_pct_calc"] = df["Current_Attendance"]
    df["missed_classes"] = ((100 - df["attendance_pct_calc"]) / 100 * 120).round()

    # ----- Academic -----
    df["fail_count"] = (df["Total_Courses"] - df["PASS"]).clip(lower=0)
    df["pass_ratio"] = df["PASS"] / df["Total_Courses"]

    # ----- Achievements (safe defaults) -----
    if "certifications_completed" not in df.columns:
        df["certifications_completed"] = 0
    if "extra_curricular_activities" not in df.columns:
        df["extra_curricular_activities"] = 0
    if "national_awards" not in df.columns:
        df["national_awards"] = 0

    # ----- Performance score -----
    df["performance_score"] = (
        df["Cgpa"] * 1.8 +
        df["PASS"] * 0.6 +
        df["certifications_completed"] * 0.8 +
        df["national_awards"] * 1.5 -
        df["fail_count"]
    )

    feature_cols = [
        "Cgpa",
        "pass_ratio",
        "attendance_pct_calc",
        "fail_count",
        "missed_classes",
        "certifications_completed",
        "extra_curricular_activities",
        "national_awards",
        "performance_score"
    ]

    X = df[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto")
    df["cluster_raw"] = kmeans.fit_predict(X_scaled)

    # ----- Rank clusters logically -----
    cluster_strength = (
        df.groupby("cluster_raw")[["Cgpa", "pass_ratio", "performance_score"]]
        .mean()
        .mean(axis=1)
        .sort_values()
    )

    cluster_rank_map = {
        old: new for new, old in enumerate(cluster_strength.index)
    }

    df["cluster"] = df["cluster_raw"].map(cluster_rank_map)

    return df, scaler, kmeans, cluster_rank_map, feature_cols


# =====================================================
# CLUSTER TEXT
# =====================================================
cluster_reviews = {
    0: "Severe academic risk. Immediate intervention required.",
    1: "Barely passing students with weak fundamentals.",
    2: "Low performers showing early improvement signs.",
    3: "Average performers lacking consistency.",
    4: "Stable mid-level academic performers.",
    5: "Good performers with growth potential.",
    6: "Strong academic students with discipline.",
    7: "High achievers with strong fundamentals.",
    8: "All-round excellent performers.",
    9: "Elite top-tier students."
}

cluster_suggestions = {
    0: "Remedial classes, counselling, mentoring.",
    1: "Concept rebuilding, structured practice.",
    2: "Attendance discipline and revision.",
    3: "Weekly assessments and mentoring.",
    4: "Skill development and certifications.",
    5: "Advanced coursework and projects.",
    6: "Internships and mentoring juniors.",
    7: "Research exposure and innovation.",
    8: "Leadership roles and national exposure.",
    9: "Research, patents, global competitions."
}

# =====================================================
# DATA UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "Upload student dataset (.xlsx)",
    type=["xlsx"]
)

if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df, scaler, kmeans, cluster_rank_map, feature_cols = train_model(df)

    # =================================================
    # DATASET PREVIEW
    # =================================================
    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # =================================================
    # CLUSTER DISTRIBUTION (CHART KEPT)
    # =================================================
    st.subheader("📊 Cluster Distribution")

    cluster_counts = df["cluster"].value_counts().sort_index()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(cluster_counts)

    with col2:
        fig, ax = plt.subplots()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax)
        ax.set_xlabel("Cluster Level")
        ax.set_ylabel("Number of Students")
        st.pyplot(fig)

    # =================================================
    # EXISTING STUDENT VIEW
    # =================================================
    st.subheader("🧍 Student Search (by Registration Number)")
    reg = st.selectbox("Select Registration Number", df["Regd No."].unique())
    student = df[df["Regd No."] == reg].iloc[0]

    st.subheader("📊 Student Academic Snapshot")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("CGPA", f"{student['Cgpa']:.2f}")
        st.metric("Attendance %", f"{student['attendance_pct_calc']:.1f}%")
        st.metric("Pass Ratio", f"{student['pass_ratio']:.2f}")

    with c2:
        st.metric("Certifications", int(student["certifications_completed"]))
        st.metric("Extra Activities", int(student["extra_curricular_activities"]))
        st.metric("National Awards", int(student["national_awards"]))

    with c3:
        st.metric("Fail Count", int(student["fail_count"]))
        st.metric("Performance Score", f"{student['performance_score']:.2f}")
        st.metric("Cluster Level", int(student["cluster"]))

    st.info(cluster_reviews[int(student["cluster"])])
    st.warning(cluster_suggestions[int(student["cluster"])])

    # =================================================
    # UNSEEN STUDENT PREDICTION
    # =================================================
    st.subheader("🆕 Predict Cluster for New Student")

    with st.form("predict_form"):
        cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0)
        total = st.number_input("Total Courses", 1, 30, 16)
        passed = st.number_input("Passed Courses", 0, total, 14)
        attendance = st.number_input("Attendance %", 0, 100, 75)
        certs = st.number_input("Certifications Completed", 0, 20, 2)
        extras = st.number_input("Extra Curricular Activities", 0, 10, 1)
        awards = st.number_input("National Awards", 0, 3, 0)

        submit = st.form_submit_button("Predict")

    if submit:
        new_student = pd.DataFrame([{
            "Cgpa": cgpa,
            "Total_Courses": total,
            "PASS": passed,
            "attendance_pct_calc": attendance,
            "missed_classes": (100 - attendance) / 100 * 120,
            "fail_count": total - passed,
            "pass_ratio": passed / total,
            "certifications_completed": certs,
            "extra_curricular_activities": extras,
            "national_awards": awards
        }])

        new_student["performance_score"] = (
            new_student["Cgpa"] * 1.8 +
            new_student["PASS"] * 0.6 +
            new_student["certifications_completed"] * 0.8 +
            new_student["national_awards"] * 1.5 -
            new_student["fail_count"]
        )

        X_new = scaler.transform(new_student[feature_cols])
        raw_cluster = kmeans.predict(X_new)[0]
        final_cluster = cluster_rank_map[raw_cluster]

        # Elite override
        if cgpa >= 9.5 and attendance >= 95 and passed == total:
            final_cluster = 9

        st.success(f"Predicted Cluster Level: {final_cluster}")
        st.info(cluster_reviews[final_cluster])
        st.warning(cluster_suggestions[final_cluster])

else:
    st.info("👆 Upload the dataset to start analysis")
