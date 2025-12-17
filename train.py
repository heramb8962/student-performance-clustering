# =========================================================
# TRAINING PIPELINE (RUN ONCE LOCALLY / COLAB)
# =========================================================

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------
# 1. LOAD DATASET (LOCAL FILE)
# ----------------------------
df = pd.read_excel("cleaned_student_dataset_random.xlsx")

# ----------------------------
# 2. FEATURE ENGINEERING
# ----------------------------
df["attendance_pct_calc"] = df["Current_Attendance"]

df["missed_classes"] = (df["Delivered"] - df["Attended"]).clip(lower=0)

df["fail_count"] = (df["Total_Courses"] - df["PASS"]).clip(lower=0)

df["pass_ratio"] = df["PASS"] / df["Total_Courses"]

df["performance_score"] = (
    df["Cgpa"] * 1.5 +
    df["PASS"] * 0.5 -
    df["fail_count"]
)

feature_cols = [
    "Cgpa",
    "Total_Courses",
    "PASS",
    "fail_count",
    "attendance_pct_calc",
    "missed_classes",
    "pass_ratio",
    "performance_score"
]

X = df[feature_cols]

# ----------------------------
# 3. SCALE + TRAIN MODEL
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto")
df["cluster_raw"] = kmeans.fit_predict(X_scaled)

# ----------------------------
# 4. CLUSTER RANKING
# ----------------------------
cluster_strength = (
    df.groupby("cluster_raw")[["Cgpa", "pass_ratio", "performance_score"]]
    .mean()
    .mean(axis=1)
    .sort_values()
)

cluster_rank_map = {old: new for new, old in enumerate(cluster_strength.index)}
df["cluster"] = df["cluster_raw"].map(cluster_rank_map)

# ----------------------------
# 5. SAVE ARTIFACTS
# ----------------------------
joblib.dump(scaler, "scaler.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(cluster_rank_map, "cluster_rank_map.pkl")

df.to_excel("student_dataset_ready_for_dashboard.xlsx", index=False)

print("✅ Training complete. Files saved.")
