import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
file_path = r"C:\Local_Git_Repository\MLAI_project\Resaleflatprices.csv"
df = pd.read_csv(file_path)
new_df=df.sample(n=50000, random_state=42)
del new_df["street_name"]
del new_df["block"]
print(new_df)
print(np.shape(new_df))
print(new_df.isna().sum())
