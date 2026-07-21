import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
file_path = r"C:\Local_Git_Repository\MLAI_project\Resaleflatprices.csv"
df = pd.read_csv(file_path)
new_df=df.sample(n=50000, random_state=42)
print(new_df)
print(np.shape(new_df))

features=["month","town", "flat_type", "block", "street_name", "storey_range", "flat_model", "lease_commence_date","remaining_lease", "resale_price"]
X=new_df[features]
y=["resale_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)