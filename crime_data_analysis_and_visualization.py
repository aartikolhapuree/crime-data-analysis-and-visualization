import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

sns.set_theme(style="whitegrid")
os.makedirs("output", exist_ok=True)

df = pd.read_csv("chicago_crimes.csv", low_memory=False)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("Rows before cleaning:", len(df))
print(df.shape)
df.head()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "primary_type"])
df = df.drop_duplicates(subset="id")
df["primary_type"] = df["primary_type"].str.strip().str.title()
df["arrest"] = df["arrest"].astype(str).str.lower().map({"true": 1, "false": 0})
df["domestic"] = df["domestic"].astype(str).str.lower().map({"true": 1, "false": 0})

print("Rows after cleaning:", len(df))
print(df[["date", "primary_type", "district"]].isna().sum())

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek      # 0 = Monday
df["hour"] = df["date"].dt.hour

def time_period(h):
    if 5 <= h < 12:  return "Morning"
    if 12 <= h < 17: return "Afternoon"
    if 17 <= h < 21: return "Evening"
    return "Night"

df["time_period"] = df["hour"].apply(time_period)
print("Period:", df["date"].min().date(), "to", df["date"].max().date())

print("Total recorded incidents:", len(df))
print("Number of categories:", df["primary_type"].nunique())
print("Arrest recorded (%):", round(df["arrest"].mean() * 100, 1))
print()
cat_table = df["primary_type"].value_counts().to_frame("count")
cat_table["percent"] = (cat_table["count"] / len(df) * 100).round(2)
cat_table.head(10)

top = df["primary_type"].value_counts().head(6)
plt.figure(figsize=(9, 5))
plt.bar(top.index, top.values, color="#1f77b4")
plt.title("Recorded incidents by crime category (top 6)")
plt.xlabel("Crime category"); plt.ylabel("Incident count")
plt.xticks(rotation=25)
plt.tight_layout(); plt.savefig("output/graph1_category.png"); plt.show()

monthly = df.groupby(df["date"].dt.to_period("M")).size()
plt.figure(figsize=(11, 4.5))
plt.plot(monthly.index.astype(str), monthly.values, marker="o", markersize=3)
plt.xticks(range(0, len(monthly), 6), monthly.index.astype(str)[::6], rotation=45)
plt.title("Monthly recorded incidents")
plt.xlabel("Month"); plt.ylabel("Incident count")
plt.tight_layout(); plt.savefig("output/graph2_monthly.png"); plt.show()

area = df["district"].dropna().astype(int).value_counts().head(5).sort_values()
plt.figure(figsize=(8, 4))
plt.barh(["District " + str(d) for d in area.index], area.values, color="#1f77b4")
plt.title("Top 5 police districts by recorded incidents (counts only)")
plt.xlabel("Incident count")
plt.tight_layout(); plt.savefig("output/graph3_area.png"); plt.show()

top_districts = df["district"].dropna().astype(int).value_counts().head(5).index
sub = df[df["district"].isin(top_districts)]
heat = pd.crosstab(sub["district"].astype(int), sub["time_period"])
heat = heat[["Morning", "Afternoon", "Evening", "Night"]]
plt.figure(figsize=(8, 4.5))
sns.heatmap(heat, annot=True, fmt="d", cmap="viridis", cbar_kws={"label": "Incident count"})
plt.title("District vs time of day (recorded incidents)")
plt.xlabel("Time of day"); plt.ylabel("District")
plt.tight_layout(); plt.savefig("output/graph4_heatmap.png"); plt.show()

ml = df.dropna(subset=["arrest", "district", "domestic"]).copy()
ml = ml.sample(n=min(200000, len(ml)), random_state=42)

top15 = ml["primary_type"].value_counts().head(15).index
ml["primary_type"] = ml["primary_type"].where(ml["primary_type"].isin(top15), "Other")

features = ["primary_type", "district", "hour", "day_of_week", "month", "domestic"]
X = pd.get_dummies(ml[features], columns=["primary_type"])
y = ml["arrest"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, max_depth=12,
                               class_weight="balanced", random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("Share of 'arrest' in test data (baseline):", round(y_test.mean(), 3))
print()
print(classification_report(y_test, pred, target_names=["No arrest", "Arrest"]))

ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                       display_labels=["No arrest", "Arrest"]).plot(cmap="Blues")
plt.title("Confusion matrix (test data)")
plt.savefig("output/graph5_confusion_matrix.png"); plt.show()

imp = pd.Series(model.feature_importances_, index=X.columns).sort_values().tail(10)
plt.figure(figsize=(8, 4.5))
imp.plot(kind="barh", color="#2a9d8f")
plt.title("Top 10 features used by the model")
plt.xlabel("Importance")
plt.tight_layout(); plt.savefig("output/graph6_feature_importance.png"); plt.show()

df.to_csv("output/crimes_clean.csv", index=False)
print("Saved output/crimes_clean.csv")