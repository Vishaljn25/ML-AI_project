
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
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


# ----------------------------------------------------- ----
# STEP 2: Drop columns we don't need
# ---------------------------------------------------------
new_df.drop(columns=[ "block", "lease_commence_date"], inplace=True)


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
# Using the IQR method: any value more than 1.5x the interquartile
# range beyond the 25th/75th percentile is flagged as an outlier.
# We report how many exist per column, but do NOT modify the data.
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
# CURRENT STATE
# ---------------------------------------------------------
print("\nCurrent cleaned dataset (pre-encoding, pre-scaling):")
print(new_df.head())
print("Current shape:", new_df.shape)

# ---------------------------------------------------------
# STEP 8: Label-encode town, flat_type, flat_model (with legend)
# ---------------------------------------------------------
# NOTE: this is label encoding, not the one-hot encoding the project
# requires — this is for readability/reference. Swap to pd.get_dummies()
# for the actual model input.

legends = {}
for col in ["town", "flat_type","street_name","flat_model"]:
    cat = new_df[col].astype("category")
    legends[col] = dict(enumerate(cat.cat.categories))
    new_df[col] = cat.cat.codes

for col, legend in legends.items():
    print(f"--- {col} legend ---")
    for code, label in legend.items():
        print(f"{code}: {label}")
    print()

    # ---------------------------------------------------------
# Display the full cleaned + label-encoded dataset
# ---------------------------------------------------------
pd.set_option("display.max_columns", None)   # show all columns, no truncation
pd.set_option("display.width", 200)          # wider display so columns don't wrap

print("\nFull cleaned dataset (all 50,000 rows, label-encoded):")
print(new_df)   # printing all 50,000 rows directly

print("\nFinal shape:", new_df.shape)



features = ["town", "flat_type","street_name","floor_area_sqm", "flat_model", "storey_mid","remaining_lease_years","sale_year","sale_month"]
X = pd.get_dummies(new_df[features], columns=["town", "flat_type","street_name", "flat_model"])

y = new_df["resale_price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, 
)





hgb_model = HistGradientBoostingRegressor(
    max_depth=15, learning_rate=0.05, max_iter=500, random_state=42
)





hgb_model.fit(X_train, y_train)
y_pred_hgb = hgb_model.predict(X_test)

print(pd.DataFrame({
    "Actual resale_prices": y_test.values,
    "Predicted resale_prices": y_pred_hgb.round(1)
}).head(5))
print("HGB R²:", round(r2_score(y_test, y_pred_hgb), 3))
print("HGB MAE:", round(mean_absolute_error(y_test, y_pred_hgb),3))


forest_model = RandomForestRegressor(n_estimators=50, max_depth=25, random_state=42)
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict(X_test)
print(pd.DataFrame({
    "Actual resale_prices": y_test.values,
    "Predicted resale_prices": y_pred_forest.round(1)
}).head(5))
print("Random Forest R²:", round(r2_score(y_test, y_pred_forest), 3))
print("Random Forest MAE:", round(mean_absolute_error(y_test, y_pred_forest), 2))



print("HGB train MAE:", round(mean_absolute_error(y_train, hgb_model.predict(X_train)),2))
print("HGB test MAE :", round(mean_absolute_error(y_test, y_pred_hgb),2))
print("Forest train MAE:", round(mean_absolute_error(y_train, forest_model.predict(X_train)), 2))
print("Forest test MAE :", round(mean_absolute_error(y_test, y_pred_forest),2))



user_input_town = input("Enter town:").strip().upper()
user_input_flat_type = input("Enter flat_type:").strip().upper()
user_input_street_name = input("Enter street_name:").strip().upper()
user_input_floor_area = float(input("Enter floor area(sqm):"))
user_input_flat_model = input("Enter flat_model:").strip()
user_input_storey_mid = float(input("Enter storey_mid:"))
user_input_remaining_lease_years = float(input("Enter remaining lease left:"))
user_input_sale_year = int(input("Enter sale_year:"))
user_input_sale_month = int(input("Enter sale_month:"))

# Build a single-row DataFrame using the RAW column names/values
# (not the quoted variable-name strings from before)
new_row_raw = pd.DataFrame([{
    "town": user_input_town,
    "flat_type": user_input_flat_type,
    "street_name": user_input_street_name,
    "floor_area_sqm": user_input_floor_area,
    "flat_model": user_input_flat_model,
    "storey_mid": user_input_storey_mid,
    "remaining_lease_years": user_input_remaining_lease_years,
    "sale_year": user_input_sale_year,
    "sale_month": user_input_sale_month,
}])

# One-hot encode this row the same way training data was encoded
new_row_encoded = pd.get_dummies(new_row_raw, columns=["town", "flat_type","street_name", "flat_model"])

# Force it to have exactly the same columns as X_train (same order,
# any dummy column not present in this input filled with 0)
new_row_encoded = new_row_encoded.reindex(columns=X_train.columns, fill_value=0)

new_y_pred_forest = forest_model.predict(new_row_encoded)

new_y_pred_hgb =  hgb_model.predict(new_row_encoded)
print("predicted resale_value for random forest:", round(new_y_pred_forest[0], 2))

print("predicted resale_value for HGB:", round(new_y_pred_hgb[0], 2))

# ===========================================================
# MODEL EVALUATION & COMPARISON  (paste this in right after
# you've computed y_pred_hgb and y_pred_forest, before the
# manual `input()` prediction section)
# ===========================================================
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1) Collect all metrics in one place (adds RMSE, which the
#    rubric explicitly asks for and your script doesn't compute yet)
# ---------------------------------------------------------
def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

hgb_train_mae, hgb_train_rmse, hgb_train_r2 = get_metrics(y_train, hgb_model.predict(X_train))
hgb_test_mae, hgb_test_rmse, hgb_test_r2   = get_metrics(y_test, y_pred_hgb)

forest_train_mae, forest_train_rmse, forest_train_r2 = get_metrics(y_train, forest_model.predict(X_train))
forest_test_mae, forest_test_rmse, forest_test_r2   = get_metrics(y_test, y_pred_forest)

print("\n=== Metrics summary ===")
print(f"HGB     -> Train MAE: {hgb_train_mae:,.0f} | Test MAE: {hgb_test_mae:,.0f} | Test RMSE: {hgb_test_rmse:,.0f} | Test R²: {hgb_test_r2:.3f}")
print(f"Forest  -> Train MAE: {forest_train_mae:,.0f} | Test MAE: {forest_test_mae:,.0f} | Test RMSE: {forest_test_rmse:,.0f} | Test R²: {forest_test_r2:.3f}")


# ---------------------------------------------------------
# 2) Actual vs Predicted scatter — visually shows which model
#    tracks the diagonal (perfect prediction line) more tightly
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
# 3) Residual plots — spots bias/heteroscedasticity.
#    A random scatter around 0 = good. A funnel/curve shape = problem.
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
# 4) Metric comparison bar chart — the core "which is better" answer
# ---------------------------------------------------------
metrics_df = pd.DataFrame({
    "Model": ["HGB", "Random Forest"],
    "MAE":  [hgb_test_mae, forest_test_mae],
    "RMSE": [hgb_test_rmse, forest_test_rmse],
    "R2":   [hgb_test_r2, forest_test_r2],
})

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, title in zip(axes, ["MAE", "RMSE", "R2"], ["MAE (lower=better)", "RMSE (lower=better)", "R² (higher=better)"]):
    ax.bar(metrics_df["Model"], metrics_df[col], color=["#4C72B0", "#DD8452"])
    ax.set_title(title)
    for i, v in enumerate(metrics_df[col]):
        ax.text(i, v, f"{v:,.2f}" if col == "R2" else f"{v:,.0f}", ha='center', va='bottom')

plt.suptitle("Model Comparison on Test Set")
plt.tight_layout()
plt.savefig("metric_comparison.png", dpi=150)
plt.show()


# ---------------------------------------------------------
# 5) Train vs Test MAE — the overfitting/underfitting check
#    Big gap (train << test) = overfitting.
#    Both high and close = underfitting.
# ---------------------------------------------------------
fit_df = pd.DataFrame({
    "Model": ["HGB", "HGB", "Random Forest", "Random Forest"],
    "Set":   ["Train", "Test", "Train", "Test"],
    "MAE":   [hgb_train_mae, hgb_test_mae, forest_train_mae, forest_test_mae],
})

fig, ax = plt.subplots(figsize=(7, 5))
width = 0.35
models = ["HGB", "Random Forest"]
train_vals = [hgb_train_mae, forest_train_mae]
test_vals = [hgb_test_mae, forest_test_mae]
x = np.arange(len(models))

ax.bar(x - width/2, train_vals, width, label="Train MAE", color="#4C72B0")
ax.bar(x + width/2, test_vals, width, label="Test MAE", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("MAE")
ax.set_title("Train vs Test MAE (Overfitting Check)")
ax.legend()
for i, (tr, te) in enumerate(zip(train_vals, test_vals)):
    ax.text(i - width/2, tr, f"{tr:,.0f}", ha='center', va='bottom', fontsize=8)
    ax.text(i + width/2, te, f"{te:,.0f}", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("overfit_check.png", dpi=150)
plt.show()


# ---------------------------------------------------------
# 6) Feature importance — supports "Model Selection & Training"
#    marks too, since it justifies which features drive price
# ---------------------------------------------------------
importances = pd.Series(forest_model.feature_importances_, index=X_train.columns)
top_15 = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 6))
top_15.sort_values().plot(kind="barh", color="#55A868")
plt.xlabel("Importance")
plt.title("Random Forest — Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()