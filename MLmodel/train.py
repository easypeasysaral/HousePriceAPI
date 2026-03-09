import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Housing.csv")

# Encode categorical features
le = LabelEncoder()

categorical_cols = [
    'mainroad','guestroom','basement',
    'hotwaterheating','airconditioning',
    'prefarea','furnishingstatus'
]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("price", axis=1)
y = df["price"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

accuracy_percentage = r2 * 100

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
print("Model Accuracy (%):", accuracy_percentage)

# Save model
pickle.dump(model, open("house_model.pkl", "wb"))