# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('insurance.csv')

# Data Preprocessing
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
scaler = StandardScaler()
df[['age', 'bmi', 'children']] = scaler.fit_transform(df[['age', 'bmi', 'children']])

# Define features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Initial Model Training and Evaluation
print("\n1. Initial Model Evaluation:")
print("--------------------------")
# Train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr.predict(X_test)

# Evaluate initial model
print("\nInitial Linear Regression Results:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# Plot Actual vs Predicted for initial Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Linear Regression: Actual vs Predicted Charges')
plt.legend()
plt.show()

# 2. Model Tuning
print("\n2. Model Tuning:")
print("---------------")
# Define parameter grid with valid parameters
param_grid = {
    'fit_intercept': [True, False],
    'positive': [True, False]
}

# Perform grid search
grid_search = GridSearchCV(
    LinearRegression(),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Print tuning results
print("\nBest parameters:", grid_search.best_params_)
print(f"Best cross-validation MSE: {-grid_search.best_score_:.4f}")

# 3. Evaluate Tuned Model
print("\n3. Tuned Model Evaluation:")
print("-------------------------")
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

print("\nTuned Linear Regression Results:")
print("MSE:", mean_squared_error(y_test, y_pred_tuned))
print("R2 Score:", r2_score(y_test, y_pred_tuned))

# Plot Actual vs Predicted for the Tuned Model
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_tuned, alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Tuned Linear Regression: Actual vs Predicted Charges')
plt.legend()
plt.show()

# Performance Summary
print("\nModel Performance Summary:")
print("-------------------------")
print("\nInitial Model:")
print(f"MSE: {mean_squared_error(y_test, y_pred_lr):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred_lr):.4f}")

print("\nTuned Model:")
print(f"MSE: {mean_squared_error(y_test, y_pred_tuned):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred_tuned):.4f}")