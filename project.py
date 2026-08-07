import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from scipy.stats import randint, uniform

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------------------------------------------------------
# STEP 1: Load the raw dataset
# ---------------------------------------------------------
file_path = r"C:\Local_Git_Repository\MLAI_project\Resaleflatprices.csv"

df = pd.read_csv(file_path)
print("Full dataset shape:", df.shape)

new_df = df.sample(n=50000, random_state=42).copy()
print("Sampled dataset shape:", new_df.shape)


# ---------------------------------------------------------
# STEP 1b: Set aside a TRUE holdout set of 10 rows that are NOT
# part of the 50,000-row sample used anywhere in training or the
# X_test/y_test split. This is a stronger real-world check than
# X_test/y_test, since those 10,000 test rows are still drawn from
# inside the same 50k sample the models were tuned against — these
# 10 rows are excluded from the raw dataset before that sampling
# even happens.
# ---------------------------------------------------------
remaining_df = df.drop(new_df.index).copy()
holdout_raw = remaining_df.sample(n=10, random_state=7).copy()
print("Holdout rows (from outside the 50k sample):", holdout_raw.shape)


# ---------------------------------------------------------
# STEP 2: Drop columns we don't need
# ---------------------------------------------------------
new_df.drop(columns=["block", "lease_commence_date"], inplace=True)


# ---------------------------------------------------------
# STEP 3: Check for missing values
# ---------------------------------------------------------
print("\nMissing values per column:")
print(new_df.isnull().sum())
print("Any missing values at all:", new_df.isnull().values.any())


# ---------------------------------------------------------
# STEP 4: storey_range -> single numeric column
# ---------------------------------------------------------
new_df["storey_low"] = new_df["storey_range"].str.split(" TO ").str[0].astype(int)
new_df["storey_high"] = new_df["storey_range"].str.split(" TO ").str[1].astype(int)
new_df["storey_mid"] = (new_df["storey_low"] + new_df["storey_high"]) / 2
new_df.drop(columns=["storey_range", "storey_low", "storey_high"], inplace=True)


# ---------------------------------------------------------
# STEP 5: remaining_lease -> single numeric column
# ---------------------------------------------------------
def parse_lease(lease_text):
    years = int(lease_text.split(" years")[0])
    months = 0
    if "months" in lease_text:
        months = int(lease_text.split(" years ")[1].split(" months")[0])
    elif "month" in lease_text:
        months = int(lease_text.split(" years ")[1].split(" month")[0])
    return years + months / 12

new_df["remaining_lease_years"] = new_df["remaining_lease"].apply(parse_lease)
new_df.drop(columns=["remaining_lease"], inplace=True)


# ---------------------------------------------------------
# STEP 6: Split "month" into sale_year and sale_month
# ---------------------------------------------------------
new_df["sale_year"] = new_df["month"].str.split("-").str[0].astype(int)
new_df["sale_month"] = new_df["month"].str.split("-").str[1].astype(int)
new_df.drop(columns=["month"], inplace=True)


# ---------------------------------------------------------
# STEP 7: Check for outliers (detection only — nothing removed)
# ---------------------------------------------------------
def check_outliers_iqr(frame, col):
    q1 = frame[col].quantile(0.25)
    q3 = frame[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = ((frame[col] < lower) | (frame[col] > upper)).sum()
    pct = n_outliers / len(frame) * 100
    print(f"{col}: {n_outliers} outliers ({pct:.2f}%) outside bounds ({lower:,.1f} to {upper:,.1f})")

print("\nOutlier check (informational only, no values changed):")
for col in ["resale_price", "floor_area_sqm", "remaining_lease_years"]:
    check_outliers_iqr(new_df, col)


# ---------------------------------------------------------
# STEP 8: Frequency-encode street_name (SPEED FIX)
# ---------------------------------------------------------
# street_name has 500+ unique values. One-hot encoding it exploded the
# column count and slowed down every model fit (especially Random Forest
# and the 150+ fits done during hyperparameter search). Frequency
# encoding keeps the signal (rarer streets vs common streets) in a
# single numeric column instead of hundreds of dummy columns.
street_freq_map = new_df["street_name"].value_counts(normalize=True)
new_df["street_name_freq"] = new_df["street_name"].map(street_freq_map)


# ---------------------------------------------------------
# STEP 9: Label-encode town, flat_type, flat_model (with legend)
# ---------------------------------------------------------
legends = {}
for col in ["town", "flat_type", "street_name", "flat_model"]:
    cat = new_df[col].astype("category")
    legends[col] = dict(enumerate(cat.cat.categories))
    new_df[col] = cat.cat.codes

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\nFinal shape:", new_df.shape)


# ---------------------------------------------------------
# STEP 10: Build features (street_name is now a frequency
# column, NOT one-hot encoded — this is the main speed win)
# ---------------------------------------------------------
features = ["town", "flat_type", "street_name_freq", "floor_area_sqm", "flat_model",
            "storey_mid", "remaining_lease_years", "sale_year", "sale_month"]

X = pd.get_dummies(new_df[features], columns=["town", "flat_type", "flat_model"])
y = new_df["resale_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)


# ===========================================================
# HYPERPARAMETER TUNING for HGB and Random Forest (SPEED FIXES)
# ===========================================================
# Speed fixes applied vs the previous version:
#  1. n_jobs=-1 removed from RandomForestRegressor itself (was fighting
#     with RandomizedSearchCV's own n_jobs=-1 for CPU cores)
#  2. cv folds reduced 5 -> 3
#  3. n_iter reduced 30 -> 15 candidates
#  4. Removed max_depth=None (unbounded trees) from both search spaces
#     — unbounded trees are the slowest candidates AND the most prone
#     to overfitting, so cutting them helps speed and quality together
#  5. Narrower n_estimators / max_iter ranges so no single candidate
#     is disproportionately slow
cv = KFold(n_splits=3, shuffle=True, random_state=42)

# ---------------------------------------------------------
# HistGradientBoostingRegressor tuning
# ---------------------------------------------------------
hgb_param_dist = {
    "max_depth": [3, 5, 7, 10],
    "max_leaf_nodes": [15, 31, 63, 127],
    "learning_rate": uniform(0.01, 0.19),
    "max_iter": [150, 250, 350],
    "l2_regularization": uniform(0, 1),
    "min_samples_leaf": [10, 20, 30, 50],
}

hgb_search = RandomizedSearchCV(
    HistGradientBoostingRegressor(random_state=42, early_stopping=True, validation_fraction=0.1),
    param_distributions=hgb_param_dist,
    n_iter=15,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=2,
)

hgb_search.fit(X_train, y_train)
print("\nBest HGB params:", hgb_search.best_params_)
print("Best HGB CV MAE:", -hgb_search.best_score_)

hgb_model = hgb_search.best_estimator_
y_pred_hgb = hgb_model.predict(X_test)


# ---------------------------------------------------------
# RandomForestRegressor tuning
# ---------------------------------------------------------
forest_param_dist = {
    "n_estimators": randint(100, 300),
    "max_depth": [5, 10, 15, 20],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 20),
    "max_features": ["sqrt", "log2", 0.5],
}

forest_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=1),   # n_jobs=1 here, search handles parallelism
    param_distributions=forest_param_dist,
    n_iter=15,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=2,
)

forest_search.fit(X_train, y_train)
print("\nBest Forest params:", forest_search.best_params_)
print("Best Forest CV MAE:", -forest_search.best_score_)

forest_model = forest_search.best_estimator_
y_pred_forest = forest_model.predict(X_test)


# ===========================================================
# MODEL EVALUATION & COMPARISON — now includes MSE everywhere
# alongside MAE, RMSE, and R²
# ===========================================================
def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

hgb_train_mae, hgb_train_mse, hgb_train_rmse, hgb_train_r2 = get_metrics(y_train, hgb_model.predict(X_train))
hgb_test_mae, hgb_test_mse, hgb_test_rmse, hgb_test_r2 = get_metrics(y_test, y_pred_hgb)

forest_train_mae, forest_train_mse, forest_train_rmse, forest_train_r2 = get_metrics(y_train, forest_model.predict(X_train))
forest_test_mae, forest_test_mse, forest_test_rmse, forest_test_r2 = get_metrics(y_test, y_pred_forest)

print("\n=== Metrics summary (Train) ===")
print(f"HGB     -> MAE: {hgb_train_mae:,.0f} | MSE: {hgb_train_mse:,.0f} | RMSE: {hgb_train_rmse:,.0f} | R²: {hgb_train_r2:.3f}")
print(f"Forest  -> MAE: {forest_train_mae:,.0f} | MSE: {forest_train_mse:,.0f} | RMSE: {forest_train_rmse:,.0f} | R²: {forest_train_r2:.3f}")

print("\n=== Metrics summary (Test) ===")
print(f"HGB     -> MAE: {hgb_test_mae:,.0f} | MSE: {hgb_test_mse:,.0f} | RMSE: {hgb_test_rmse:,.0f} | R²: {hgb_test_r2:.3f}")
print(f"Forest  -> MAE: {forest_test_mae:,.0f} | MSE: {forest_test_mse:,.0f} | RMSE: {forest_test_rmse:,.0f} | R²: {forest_test_r2:.3f}")


# ---------------------------------------------------------
# Actual vs Predicted scatter
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

for ax, y_pred, name in zip(axes, [y_pred_hgb, y_pred_forest], ["HistGradientBoosting", "Random Forest"]):
    ax.scatter(y_test, y_pred, alpha=0.3, s=10)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual resale_price")
    ax.set_ylabel("Predicted resale_price")
    ax.set_title(name)
    ax.legend()

plt.suptitle("Actual vs Predicted Resale Price")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()


# ---------------------------------------------------------
# Residual plots
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

for ax, y_pred, name in zip(axes, [y_pred_hgb, y_pred_forest], ["HistGradientBoosting", "Random Forest"]):
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax.axhline(0, color='r', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Predicted resale_price")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(name)

plt.suptitle("Residual Plots")
plt.tight_layout()
plt.savefig("residuals.png", dpi=150)
plt.show()


# ---------------------------------------------------------
# Metric comparison bar chart — now 4 panels: MAE, MSE, RMSE, R2
# ---------------------------------------------------------
metrics_df = pd.DataFrame({
    "Model": ["HGB", "Random Forest"],
    "MAE":  [hgb_test_mae, forest_test_mae],
    "MSE":  [hgb_test_mse, forest_test_mse],
    "RMSE": [hgb_test_rmse, forest_test_rmse],
    "R2":   [hgb_test_r2, forest_test_r2],
})

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
panel_specs = [
    ("MAE", "MAE (lower=better)"),
    ("MSE", "MSE (lower=better)"),
    ("RMSE", "RMSE (lower=better)"),
    ("R2", "R² (higher=better)"),
]
bar_width = 0.45
x_pos = [0, 1]

for ax, (col, title) in zip(axes, panel_specs):
    ax.bar(x_pos, metrics_df[col], width=bar_width, color=["#4C72B0", "#DD8452"], edgecolor="none")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_df["Model"])
    ax.set_xlim(-0.9, 1.9)   # extra padding on both sides so bars don't fill the panel
    ax.set_title(title, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, v in enumerate(metrics_df[col]):
        label = f"{v:,.2f}" if col == "R2" else f"{v:,.0f}"
        ax.text(i, v, label, ha='center', va='bottom', fontsize=8)

plt.suptitle("Model Comparison on Test Set")
plt.tight_layout()
plt.savefig("metric_comparison.png", dpi=150)
plt.show()


# ---------------------------------------------------------
# Train vs Test MAE — overfitting/underfitting check
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
width = 0.28
models = ["HGB", "Random Forest"]
train_vals = [hgb_train_mae, forest_train_mae]
test_vals = [hgb_test_mae, forest_test_mae]
x = np.arange(len(models))

ax.bar(x - width/2 - 0.02, train_vals, width, label="Train MAE", color="#4C72B0", edgecolor="none")
ax.bar(x + width/2 + 0.02, test_vals, width, label="Test MAE", color="#DD8452", edgecolor="none")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_xlim(-0.7, 1.7)
ax.set_ylabel("MAE")
ax.set_title("Train vs Test MAE (Overfitting Check)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)
for i, (tr, te) in enumerate(zip(train_vals, test_vals)):
    ax.text(i - width/2 - 0.02, tr, f"{tr:,.0f}", ha='center', va='bottom', fontsize=8)
    ax.text(i + width/2 + 0.02, te, f"{te:,.0f}", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("overfit_check.png", dpi=150)
plt.show()


# ---------------------------------------------------------
# Feature importance — grouped back to the ORIGINAL features
# (town, flat_type, street_name_freq, floor_area_sqm, flat_model,
# storey_mid, remaining_lease_years, sale_year, sale_month)
# ---------------------------------------------------------
# town, flat_type, and flat_model were one-hot encoded into many dummy
# columns (e.g. "town_ANG MO KIO", "town_BEDOK", ...). Permuting those
# dummy columns one at a time (as before) only tells you the importance
# of a single category, not the feature as a whole. Instead, this
# shuffles ALL dummy columns belonging to one original feature together
# (same row order across the group), which measures how much the model
# relies on that original feature as a whole. Works for HGB and Random
# Forest identically since it doesn't depend on either model's internals.
original_features = [
    "town", "flat_type", "street_name_freq", "floor_area_sqm", "flat_model",
    "storey_mid", "remaining_lease_years", "sale_year", "sale_month",
]

# Map each original feature name to the actual column(s) it expanded
# into in X_test (one-hot columns are named "<feature>_<category>")
feature_groups = {
    feat: [c for c in X_test.columns if c == feat or c.startswith(feat + "_")]
    for feat in original_features
}

def grouped_permutation_importance(model, X, y, groups, n_repeats=5, random_state=42):
    rng = np.random.RandomState(random_state)
    baseline_mae = mean_absolute_error(y, model.predict(X))
    scores = {}
    for feat_name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            perm_order = rng.permutation(len(X))
            X_shuffled[cols] = X_shuffled[cols].values[perm_order]
            shuffled_mae = mean_absolute_error(y, model.predict(X_shuffled))
            drops.append(shuffled_mae - baseline_mae)   # positive = feature matters (error got worse)
        scores[feat_name] = np.mean(drops)
    return scores

hgb_grouped_importance = grouped_permutation_importance(hgb_model, X_test, y_test, feature_groups)
forest_grouped_importance = grouped_permutation_importance(forest_model, X_test, y_test, feature_groups)

importance_df = pd.DataFrame({
    "Feature": original_features,
    "HGB": [hgb_grouped_importance[f] for f in original_features],
    "Random Forest": [forest_grouped_importance[f] for f in original_features],
})
importance_df["max_importance"] = importance_df[["HGB", "Random Forest"]].max(axis=1)
importance_df = importance_df.sort_values("max_importance")  # ascending, for horizontal bar order

y_pos = np.arange(len(importance_df))
bar_height = 0.32

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(y_pos - bar_height/2, importance_df["HGB"], bar_height, label="HGB", color="#4C72B0", edgecolor="none")
ax.barh(y_pos + bar_height/2, importance_df["Random Forest"], bar_height, label="Random Forest", color="#DD8452", edgecolor="none")
ax.set_yticks(y_pos)
ax.set_yticklabels(importance_df["Feature"])
ax.set_xlabel("Increase in Test MAE When Shuffled (higher = more important)")
ax.set_title("Feature Importance — HGB vs Random Forest (Original Features)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()


# ===========================================================
# TRUE HOLDOUT VALIDATION — 10 real rows from OUTSIDE the 50k
# sample entirely (set aside back in Step 1b, before sampling)
# ===========================================================
# These rows never touched X_train, X_test, y_train, y_test, or the
# hyperparameter search in any way — they're the closest thing to
# genuinely new real-world data the model hasn't seen.
def preprocess_raw_rows(raw_slice, street_freq_lookup):
    d = raw_slice.copy()

    d["storey_low"] = d["storey_range"].str.split(" TO ").str[0].astype(int)
    d["storey_high"] = d["storey_range"].str.split(" TO ").str[1].astype(int)
    d["storey_mid"] = (d["storey_low"] + d["storey_high"]) / 2

    d["remaining_lease_years"] = d["remaining_lease"].apply(parse_lease)

    d["sale_year"] = d["month"].str.split("-").str[0].astype(int)
    d["sale_month"] = d["month"].str.split("-").str[1].astype(int)

    # unseen streets (not present in the 50k training sample) fall
    # back to a frequency of 0, same as the manual-input section does
    d["street_name_freq"] = d["street_name"].map(street_freq_lookup).fillna(0)

    d["town"] = d["town"].str.strip().str.upper()
    d["flat_type"] = d["flat_type"].str.strip().str.upper()

    return d[["town", "flat_type", "street_name_freq", "floor_area_sqm", "flat_model",
              "storey_mid", "remaining_lease_years", "sale_year", "sale_month"]]

holdout_actual = holdout_raw["resale_price"].values
holdout_features = preprocess_raw_rows(holdout_raw, street_freq_map)

holdout_encoded = pd.get_dummies(holdout_features, columns=["town", "flat_type", "flat_model"])
holdout_encoded = holdout_encoded.reindex(columns=X_train.columns, fill_value=0)

holdout_pred_forest = forest_model.predict(holdout_encoded)
holdout_pred_hgb = hgb_model.predict(holdout_encoded)

holdout_mae_forest = mean_absolute_error(holdout_actual, holdout_pred_forest)
holdout_mae_hgb = mean_absolute_error(holdout_actual, holdout_pred_hgb)

print("\n=== TRUE holdout validation (10 rows never seen during training) ===")
for i in range(len(holdout_actual)):
    print(f"Row {i+1}: Actual={holdout_actual[i]:,.0f} | "
          f"HGB={holdout_pred_hgb[i]:,.0f} (err {holdout_pred_hgb[i]-holdout_actual[i]:+,.0f}) | "
          f"Forest={holdout_pred_forest[i]:,.0f} (err {holdout_pred_forest[i]-holdout_actual[i]:+,.0f})")
print(f"\nHoldout MAE -> HGB: {holdout_mae_hgb:,.0f} | Forest: {holdout_mae_forest:,.0f}")
# (chart for these 10 rows is produced in the "10 REAL SAMPLES" section below,
# which now plots this same holdout_actual / holdout_pred_hgb / holdout_pred_forest data)


# ---------------------------------------------------------
# 10 REAL SAMPLES: Actual vs HGB vs Random Forest
# (now using the TRUE holdout rows from outside the 50k sample,
# instead of a random draw from y_test)
# ---------------------------------------------------------
actual_vals = holdout_actual
hgb_vals = holdout_pred_hgb
forest_vals = holdout_pred_forest

labels = [f"Sample {i+1}" for i in range(len(actual_vals))]

print("\n=== 10 sample comparison (true holdout) ===")
for i in range(len(actual_vals)):
    print(f"{labels[i]}: Actual={actual_vals[i]:,.0f} | HGB={hgb_vals[i]:,.0f} "
          f"(err {hgb_vals[i]-actual_vals[i]:+,.0f}) | Forest={forest_vals[i]:,.0f} "
          f"(err {forest_vals[i]-actual_vals[i]:+,.0f})")

x2 = np.arange(len(labels))
width2 = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
bars_actual = ax.bar(x2 - width2, actual_vals, width2, label="Actual", color="#555555", edgecolor="none")
bars_hgb = ax.bar(x2, hgb_vals, width2, label="HGB Predicted", color="#4C72B0", edgecolor="none")
bars_forest = ax.bar(x2 + width2, forest_vals, width2, label="Random Forest Predicted", color="#DD8452", edgecolor="none")

ax.set_xticks(x2)
ax.set_xticklabels(labels)
ax.set_ylabel("Resale Price (SGD)")
ax.set_title("Actual vs Predicted Resale Price — 10 True Holdout Samples")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)

for bars in [bars_actual, bars_hgb, bars_forest]:
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h, f"{h:,.0f}",
                ha='center', va='bottom', fontsize=8, rotation=90)

plt.tight_layout()
plt.savefig("ten_sample_comparison.png", dpi=150)
plt.show()


# ---------------------------------------------------------
# % error per sample per model (true holdout)
# ---------------------------------------------------------
hgb_pct_err = (hgb_vals - actual_vals) / actual_vals * 100
forest_pct_err = (forest_vals - actual_vals) / actual_vals * 100

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x2 - width2/2, hgb_pct_err, width2, label="HGB % error", color="#4C72B0", edgecolor="none")
ax.bar(x2 + width2/2, forest_pct_err, width2, label="Random Forest % error", color="#DD8452", edgecolor="none")
ax.axhline(0, color='black', linewidth=1)
ax.set_xticks(x2)
ax.set_xticklabels(labels)
ax.set_ylabel("% Error (Predicted - Actual) / Actual")
ax.set_title("Prediction Error % — 10 True Holdout Samples")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("ten_sample_pct_error.png", dpi=150)
plt.show()


# ---------------------------------------------------------
# 10 holdout samples as a TABLE — Actual, Predicted, and % Error
# for both models, one row per sample
# ---------------------------------------------------------
hgb_pct_err_signed = (hgb_vals - actual_vals) / actual_vals * 100
forest_pct_err_signed = (forest_vals - actual_vals) / actual_vals * 100

table_df = pd.DataFrame({
    "Sample": labels,
    "Actual": [f"{v:,.0f}" for v in actual_vals],
    "HGB Pred": [f"{v:,.0f}" for v in hgb_vals],
    "HGB % Off": [f"{v:+.1f}%" for v in hgb_pct_err_signed],
    "Forest Pred": [f"{v:,.0f}" for v in forest_vals],
    "Forest % Off": [f"{v:+.1f}%" for v in forest_pct_err_signed],
})

HGB_WIN_COLOR = "#D5F5D0"      # light green — HGB closer to actual this row
FOREST_WIN_COLOR = "#D6E8FA"   # light blue — Random Forest closer to actual this row
TIE_COLOR = "#F2F2F2"          # light gray — exact tie (rare)

fig, ax = plt.subplots(figsize=(11, 1.1 + 0.45 * len(table_df)))
ax.axis("off")

tbl = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)

# Style header row
for col_idx in range(len(table_df.columns)):
    header_cell = tbl[0, col_idx]
    header_cell.set_facecolor("#4C72B0")
    header_cell.set_text_props(color="white", weight="bold")

# Color each ROW by whichever model was more accurate (smaller absolute
# % error) for that specific sample — green row = HGB won, blue row =
# Random Forest won
for row_idx in range(1, len(table_df) + 1):
    hgb_abs_off = abs(hgb_pct_err_signed[row_idx - 1])
    forest_abs_off = abs(forest_pct_err_signed[row_idx - 1])

    if hgb_abs_off < forest_abs_off:
        row_color = HGB_WIN_COLOR
    elif forest_abs_off < hgb_abs_off:
        row_color = FOREST_WIN_COLOR
    else:
        row_color = TIE_COLOR

    for col_idx in range(len(table_df.columns)):
        tbl[row_idx, col_idx].set_facecolor(row_color)

# Legend explaining the row colors, placed just under the table
legend_handles = [
    Patch(facecolor=HGB_WIN_COLOR, edgecolor="black", label="HGB more accurate this row"),
    Patch(facecolor=FOREST_WIN_COLOR, edgecolor="black", label="Random Forest more accurate this row"),
]
ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=2,
    frameon=False,
    fontsize=9,
)

plt.title("10 True Holdout Samples — Actual vs Predicted (% Off)", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig("ten_sample_table.png", dpi=150, bbox_inches="tight")
plt.show()


# ===========================================================
# MANUAL PREDICTION FROM USER INPUT
# Supports entering MULTIPLE flats at once: separate each value
# with a comma. Position 1 in every field = flat 1, position 2 =
# flat 2, etc. e.g.
#   Enter town(s): ANG MO KIO, BEDOK, TAMPINES
#   Enter floor area(s): 65, 90, 110
# predicts 3 separate flats in one go. A single value with no
# comma still works exactly as before (predicts just one flat).
# ===========================================================
def parse_str_list(prompt, upper=False):
    raw = input(prompt)
    items = [item.strip() for item in raw.split(",")]
    if upper:
        items = [item.upper() for item in items]
    return items

def parse_float_list(prompt):
    raw = input(prompt)
    return [float(item.strip()) for item in raw.split(",")]

def parse_int_list(prompt):
    raw = input(prompt)
    return [int(item.strip()) for item in raw.split(",")]

towns = parse_str_list("Enter town(s), comma-separated for multiple: ", upper=True)
flat_types = parse_str_list("Enter flat_type(s), comma-separated for multiple: ", upper=True)
street_names = parse_str_list("Enter street_name(s), comma-separated for multiple: ", upper=True)
floor_areas = parse_float_list("Enter floor area(s) sqm, comma-separated for multiple: ")
flat_models = parse_str_list("Enter flat_model(s), comma-separated for multiple: ")
storey_mids = parse_float_list("Enter storey_mid(s), comma-separated for multiple: ")
remaining_leases = parse_float_list("Enter remaining lease years left, comma-separated for multiple: ")
sale_years = parse_int_list("Enter sale_year(s), comma-separated for multiple: ")
sale_months = parse_int_list("Enter sale_month(s), comma-separated for multiple: ")

# Make sure every field has the same number of comma-separated values
lengths = {
    "town": len(towns), "flat_type": len(flat_types), "street_name": len(street_names),
    "floor_area": len(floor_areas), "flat_model": len(flat_models), "storey_mid": len(storey_mids),
    "remaining_lease_years": len(remaining_leases), "sale_year": len(sale_years), "sale_month": len(sale_months),
}
if len(set(lengths.values())) != 1:
    raise ValueError(
        f"Each field must have the same number of comma-separated values. "
        f"Got these counts per field: {lengths}"
    )

n_rows = len(towns)

# Convert each typed street name into the same frequency value used in
# training. Unseen/unknown streets fall back to 0 (rarest possible).
rows = []
for i in range(n_rows):
    rows.append({
        "town": towns[i],
        "flat_type": flat_types[i],
        "street_name_freq": street_freq_map.get(street_names[i], 0),
        "floor_area_sqm": floor_areas[i],
        "flat_model": flat_models[i],
        "storey_mid": storey_mids[i],
        "remaining_lease_years": remaining_leases[i],
        "sale_year": sale_years[i],
        "sale_month": sale_months[i],
    })

new_rows_raw = pd.DataFrame(rows)
new_rows_encoded = pd.get_dummies(new_rows_raw, columns=["town", "flat_type", "flat_model"])
new_rows_encoded = new_rows_encoded.reindex(columns=X_train.columns, fill_value=0)

new_y_pred_forest = forest_model.predict(new_rows_encoded)
new_y_pred_hgb = hgb_model.predict(new_rows_encoded)

print("\n=== Predictions ===")
for i in range(n_rows):
    print(f"Flat {i+1} ({towns[i]}, {flat_types[i]}, {floor_areas[i]} sqm): "
          f"Random Forest = {round(new_y_pred_forest[i], 2):,} | HGB = {round(new_y_pred_hgb[i], 2):,}")