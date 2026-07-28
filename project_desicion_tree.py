
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestRegressor
# ---------------------------------------------------------
# STEP 1: Load the raw dataset
# ---------------------------------------------------------
file_path = r"C:\Local_Git_Repository\MLAI_project\Resaleflatprices.csv"

df = pd.read_csv(file_path)
print("Full dataset shape:", df.shape)

new_df = df.sample(n=100000, random_state=42).copy()
print("Sampled dataset shape:", new_df.shape)


# ----------------------------------------------------- ----
# STEP 2: Drop columns we don't need
# ---------------------------------------------------------
new_df.drop(columns=["street_name", "block", "lease_commence_date"], inplace=True)


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
for col in ["town", "flat_type", "flat_model"]:
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



features = ["town", "flat_type", "floor_area_sqm", "flat_model", "storey_mid","remaining_lease_years","sale_year","sale_month"]
X = pd.get_dummies(new_df[features], columns=["town", "flat_type", "flat_model"])

y = new_df["resale_price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, 
)

tree_model = DecisionTreeRegressor(max_depth=20, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

print(pd.DataFrame({
    "Actual resale_prices": y_test.values,
    "Predicted resale_prices": y_pred_tree.round(1)
}).head(10))

mae = mean_absolute_error(y_test, y_pred_tree)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_tree))
r2 = r2_score(y_test, y_pred_tree)

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 2))
forest_model = RandomForestRegressor(n_estimators=300, max_depth=25, random_state=42)
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict(X_test)
print(pd.DataFrame({
    "Actual resale_prices": y_test.values,
    "Predicted resale_prices": y_pred_forest.round(1)
}).head(10))
print("Random Forest R²:", round(r2_score(y_test, y_pred_forest), 3))
print("Random Forest MAE:", round(mean_absolute_error(y_test, y_pred_forest), 2))


print("Tree train MAE:", mean_absolute_error(y_train, tree_model.predict(X_train)))
print("Tree test MAE :", mae)

print("Forest train MAE:", mean_absolute_error(y_train, forest_model.predict(X_train)))
print("Forest test MAE :", mean_absolute_error(y_test, y_pred_forest))
tree_param_grid = {
    "max_depth": [5, 8, 10, 12, 15, 20],
    "min_samples_leaf": [1, 5, 10, 20, 50],
    "min_samples_split": [2, 10, 20, 50]
}
tree_grid = GridSearchCV(DecisionTreeRegressor(random_state=42), tree_param_grid,
                          cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
tree_grid.fit(X_train, y_train)
print("Best params:", tree_grid.best_params_)