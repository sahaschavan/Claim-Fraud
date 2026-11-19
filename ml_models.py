import sqlite3
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
BASE = Path(__file__).resolve().parent
out_dir = BASE / "outputs"
fig_dir = out_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

con = sqlite3.connect(BASE / "data" / "claims.db")
df = pd.read_sql_query("SELECT * FROM claims", con)
con.close()

drop_cols = ["claim_id", "policy_number", "insured_zip", "incident_location",
             "policy_bind_date", "incident_date", "auto_model",
             "injury_claim", "property_claim", "vehicle_claim",
             "fraud_reported", "fraud"]
X = df.drop(columns=drop_cols)
y = df["fraud"]
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
print(f"{len(num_cols)} numeric features, {len(cat_cols)} categorical features")
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "SVM": SVC(class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
}

baseline = 1 - y.mean()
print(f"\nbaseline accuracy (always predict not fraud): {baseline:.3f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    scores = cross_validate(pipe, X, y, cv=cv,
                            scoring=["accuracy", "recall", "f1", "roc_auc"])
    rows.append({
        "model": name,
        "accuracy": scores["test_accuracy"].mean(),
        "recall": scores["test_recall"].mean(),
        "f1": scores["test_f1"].mean(),
        "roc_auc": scores["test_roc_auc"].mean(),
        "roc_auc_std": scores["test_roc_auc"].std(),
    })

results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).round(3)
print("\n5-fold CV results:")
print(results.to_string(index=False))
results.to_csv(out_dir / "model_comparison.csv", index=False)

plt.figure(figsize=(7, 4))
r = results.sort_values("roc_auc")
plt.barh(r["model"], r["roc_auc"], xerr=r["roc_auc_std"], color="#2a78d6")
plt.axvline(0.5, color="gray", linestyle="--")
plt.xlabel("ROC-AUC (5-fold CV)")
plt.title("Model comparison")
plt.tight_layout()
plt.savefig(fig_dir / "05_model_comparison.png", dpi=150)
plt.close()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

rf_pipe = Pipeline([("prep", preprocess),
                    ("model", RandomForestClassifier(class_weight="balanced", random_state=SEED))])
param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [4, 6, 10, None],
    "model__min_samples_leaf": [1, 3, 5],
}
grid = GridSearchCV(rf_pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
print("\nbest random forest params:", grid.best_params_)
print(f"best CV ROC-AUC on training data: {grid.best_score_:.3f}")

best_rf = grid.best_estimator_
pred = best_rf.predict(X_test)
proba = best_rf.predict_proba(X_test)[:, 1]
print(f"\nTEST SET ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
print(classification_report(y_test, pred, target_names=["not fraud", "fraud"]))

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["not fraud", "fraud"], cmap="Blues")
plt.title("Random forest: confusion matrix (test set)")
plt.tight_layout()
plt.savefig(fig_dir / "06_confusion_matrix.png", dpi=150)
plt.close()

names = best_rf.named_steps["prep"].get_feature_names_out()
imp = pd.Series(best_rf.named_steps["model"].feature_importances_, index=names)
imp.index = [n.replace("num__", "").replace("cat__", "") for n in imp.index]
top = imp.sort_values(ascending=False).head(12)
print("\ntop features:")
print(top.round(3).to_string())
top.to_csv(out_dir / "top_features.csv", header=["importance"])

plt.figure(figsize=(7, 5))
sns.barplot(x=top.values, y=top.index, color="#2a78d6")
plt.xlabel("Random forest feature importance")
plt.ylabel("")
plt.title("Top features for predicting fraud")
plt.tight_layout()
plt.savefig(fig_dir / "07_feature_importance.png", dpi=150)
plt.close()

cluster_cols = ["age", "months_as_customer", "policy_annual_premium", "total_claim_amount"]
Z = StandardScaler().fit_transform(df[cluster_cols])
inertias, sils = [], []
ks = range(2, 9)
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Z, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(list(ks), inertias, marker="o")
ax[0].set_title("Elbow plot")
ax[0].set_xlabel("k")
ax[0].set_ylabel("Inertia")
ax[1].plot(list(ks), sils, marker="o", color="#eb6834")
ax[1].set_title("Silhouette score")
ax[1].set_xlabel("k")
plt.tight_layout()
plt.savefig(fig_dir / "08_kmeans_choose_k.png", dpi=150)
plt.close()

sil_by_k = {k: float(s) for k, s in zip(ks, sils)}
best_k = max(range(3, 7), key=lambda k: sil_by_k[k])
print("\nsilhouette scores:", {k: round(s, 3) for k, s in sil_by_k.items()})
print("using k =", best_k)

df["cluster"] = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit_predict(Z)
profile = df.groupby("cluster").agg(
    claims=("claim_id", "count"),
    avg_age=("age", "mean"),
    avg_months_customer=("months_as_customer", "mean"),
    avg_premium=("policy_annual_premium", "mean"),
    avg_claim=("total_claim_amount", "mean"),
    fraud_rate=("fraud", "mean"),
).round(2)
print("\ncluster profiles:")
print(profile.to_string())
profile.to_csv(out_dir / "cluster_profiles.csv")
