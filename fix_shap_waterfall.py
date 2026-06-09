"""
Fix the SHAP waterfall chart — the original was generated with show=True,
which caused plt.title() / plt.tight_layout() / plt.savefig() to operate on a
stale/empty figure.  This script reproduces the exact pipeline from the
notebook and saves a correct waterfall plot.

Usage:
    python fix_shap_waterfall.py
Output:
    figures/shap_waterfall.png   (overwritten with correct chart)
    figures/shap_waterfall_fixed.png  (backup copy)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — critical
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")

print("=" * 60)
print("1. Loading data …")
df = pd.read_csv("data/bank-additional-full.csv", sep=";")
print(f"   Shape: {df.shape}")

# ── Preprocessing (identical to notebook cells 24–27) ──────────────
print("2. Preprocessing & feature engineering …")

df_proc = df.copy()
df_proc["y_encoded"] = df_proc["y"].map({"yes": 1, "no": 0})

binary_cols = ["default", "housing", "loan"]
multi_cat_cols = [
    "job", "marital", "education", "contact", "month",
    "day_of_week", "poutcome",
]
econ_cols = [
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]
num_cols = ["age", "duration", "campaign", "pdays", "previous"]

for col in binary_cols:
    df_proc[col + "_bin"] = df_proc[col].map({"yes": 1, "no": 0}).fillna(0).astype(int)

# Engineered features
df_proc["pdays_was_contacted"] = (df_proc["pdays"] != 999).astype(int)
df_proc["engagement_score"] = (df_proc["previous"] + 1) / (df_proc["campaign"] + 1)
df_proc["loan_burden"] = df_proc["housing_bin"] + df_proc["loan_bin"]
df_proc["euribor_cpi_spread"] = df_proc["euribor3m"] - (df_proc["cons.price.idx"] / 100)

engineered_features = [
    "pdays_was_contacted", "engagement_score",
    "loan_burden", "euribor_cpi_spread",
]

binary_encoded_cols = ["default_bin", "housing_bin", "loan_bin"]
production_num_cols = (
    ["age", "campaign", "pdays", "previous"]
    + econ_cols
    + engineered_features
)

y = df_proc["y_encoded"]

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
     multi_cat_cols),
    ("scaler", StandardScaler(), production_num_cols),
    ("binary", "passthrough", binary_encoded_cols),
])

X = pd.concat([
    df_proc[multi_cat_cols],
    df_proc[production_num_cols],
    df_proc[binary_encoded_cols],
], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

# Feature names (for waterfall label readability, abbreviate long names)
ohe = preprocessor.named_transformers_["onehot"]
cat_names = list(ohe.get_feature_names_out(multi_cat_cols))
all_feature_names = cat_names + production_num_cols + binary_encoded_cols

# SMOTE
smote = SMOTE(random_state=42)
resampled = smote.fit_resample(X_train_p, y_train)
X_train_smote, y_train_smote = resampled[0], resampled[1]

print(f"   Train after SMOTE: {X_train_smote.shape}, "
      f"class dist: {dict(zip(*np.unique(y_train_smote, return_counts=True)))}")

# ── Train XGBoost (identical to notebook) ─────────────────────────
print("3. Training XGBoost …")
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    random_state=42, eval_metric="logloss", verbosity=0,
)
model.fit(X_train_smote, y_train_smote)
print("   Done.")

# ── SHAP Waterfall — PROPER figure handling ───────────────────────
print("4. Computing SHAP …")
explainer = shap.TreeExplainer(model)

# Find a correctly-predicted "yes" sample (same logic as notebook)
y_pred_test = model.predict(X_test_p)
y_prob_test = model.predict_proba(X_test_p)[:, 1]
yes_mask = (y_test.values == 1) & (y_pred_test == 1)

if not yes_mask.any():
    print("   WARNING: no correctly-predicted 'yes' samples — falling back to any 'yes'")
    yes_mask = y_test.values == 1

sample_idx = np.where(yes_mask)[0][0]
print(f"   Using test sample idx={sample_idx} "
      f"(true=Yes, pred=Yes, prob={y_prob_test[sample_idx]:.3f})")

# explain the single sample
sample_exp = explainer(
    X_test_p[sample_idx : sample_idx + 1], check_additivity=False
)

# ── CRITICAL FIX ──────────────────────────────────────────────────
# shap.waterfall_plot creates ITS OWN figure internally.
# Use show=False to avoid premature plt.show(), then work with the
# figure that SHAP created via plt.gcf().
# ───────────────────────────────────────────────────────────────────
shap.waterfall_plot(
    shap.Explanation(
        values=sample_exp.values[0, :],
        base_values=sample_exp.base_values[0],
        data=X_test_p[sample_idx],
        feature_names=all_feature_names,
    ),
    max_display=12,
    show=False,                            # KEY: do NOT show yet
)

# Get the figure that shap.waterfall_plot created internally
fig = plt.gcf()
fig.set_size_inches(14, 9)

plt.title(
    'SHAP Waterfall - Correctly Predicted "Subscriber"\n'
    f"True=Yes, Pred=Yes, Prob={y_prob_test[sample_idx]:.3f}",
    fontsize=13, fontweight="bold",
)
plt.tight_layout(pad=1.5)
plt.savefig("figures/shap_waterfall.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.savefig("figures/shap_waterfall_fixed.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)

print("Done - waterfall chart regenerated with correct figure handling.")
