
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV

# ---------------------------------------------------------
# STEP 1: Load the raw dataset
# ---------------------------------------------------------
file_path = r"C:\Users\User 1\Downloads\Resaleflatprices.csv"
df = pd.read_csv(file_path)
print("Full dataset shape:", df.shape)

# The full dataset has 235,000+ rows. Working with all of it makes
# every test run slow, so we take a random 50,000-row sample instead.
# random_state=42 just means: if we run this again, we get the SAME
# random sample, not a different one each time (this keeps results reproducible).
new_df = df.sample(n=50000, random_state=42).copy()
print("Sampled dataset shape:", new_df.shape)


# ---------------------------------------------------------
# STEP 2: Drop columns we don't need
# ---------------------------------------------------------
# street_name and block are too specific (almost every flat has a unique
# combination) to help a model find general patterns, so we remove them.
new_df.drop(columns=["street_name", "block"], inplace=True)


# ---------------------------------------------------------
# STEP 3: Turn "storey_range" into a single number
# ---------------------------------------------------------
# Right now storey_range looks like the TEXT "04 TO 06".
# A model can only work with numbers, so we need to convert this.
# We split it into a low floor (4) and a high floor (6), then take
# the midpoint (5) as one representative number for "which floor".

new_df["storey_low"] = new_df["storey_range"].str.split(" TO ").str[0].astype(int)
new_df["storey_high"] = new_df["storey_range"].str.split(" TO ").str[1].astype(int)
new_df["storey_mid"] = (new_df["storey_low"] + new_df["storey_high"]) / 2

# We only needed storey_mid, so we can drop the original text column
# and the two helper columns we used to build it.
new_df.drop(columns=["storey_range", "storey_low", "storey_high"], inplace=True)

print("\nExample of storey conversion:")
print(new_df[["storey_mid"]].head())


# ---------------------------------------------------------
# STEP 4: Turn "remaining_lease" into a single number
# ---------------------------------------------------------
# remaining_lease looks like the TEXT "61 years 04 months".
# We write a small function that reads that text and converts it
# into a decimal number of years, e.g. "61 years 04 months" -> 61.33

def parse_lease(lease_text):
    # Everything before " years" is the whole number of years
    years = int(lease_text.split(" years")[0])

    # Some rows say "X months" (plural) and some say "X month" (singular).
    # We check for both so the function doesn't crash on either format.
    months = 0
    if "months" in lease_text:
        months = int(lease_text.split(" years ")[1].split(" months")[0])
    elif "month" in lease_text:
        months = int(lease_text.split(" years ")[1].split(" month")[0])

    # Convert months into a fraction of a year and add it on
    return years + months / 12

new_df["remaining_lease_years"] = new_df["remaining_lease"].apply(parse_lease)
new_df.drop(columns=["remaining_lease"], inplace=True)

print("\nExample of lease conversion:")
print(new_df[["remaining_lease_years"]].head())


# ---------------------------------------------------------
# STEP 5: Split "month" into separate year and month numbers
# ---------------------------------------------------------
# month looks like the TEXT "2017-01" (year-month).
# Splitting it lets the model consider year and month as SEPARATE
# numeric clues — e.g. prices generally rise over the years, and
# there might be seasonal patterns within a year too.

new_df["sale_year"] = new_df["month"].str.split("-").str[0].astype(int)
new_df["sale_month"] = new_df["month"].str.split("-").str[1].astype(int)
new_df.drop(columns=["month"], inplace=True)

print("\nExample of month conversion:")
print(new_df[["sale_year", "sale_month"]].head())


# ---------------------------------------------------------
# STEP 6: Convert category columns into numbers (label encoding)
# ---------------------------------------------------------
# town, flat_type, and flat_model are all TEXT categories
# (e.g. "ANG MO KIO", "3 ROOM", "Improved"). A model needs numbers,
# so we assign each unique category its own number code.
# Example: town might become ANG MO KIO -> 0, BEDOK -> 1, BISHAN -> 2, etc.

for col in ["town", "flat_type", "flat_model"]:
    new_df[col] = new_df[col].astype("category").cat.codes

print("\nExample of category encoding:")
print(new_df[["town", "flat_type", "flat_model"]].head())


# ---------------------------------------------------------
# FINAL RESULT: fully numeric, ready for modelling
# ---------------------------------------------------------
print("\nFinal cleaned dataset:")
print(new_df.head())
print("Final shape:", np.shape(new_df))