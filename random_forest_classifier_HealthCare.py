import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("Loading and preparing data...")
df = pd.read_csv('healthcare_dataset.csv')

# Drop non-predictive columns
columns_to_drop = ['Name', 'Doctor', 'Room Number', 'Date of Admission', 
                  'Discharge Date', 'Hospital']
df = df.drop(columns=columns_to_drop)

# Convert categorical variables to numerical
le = LabelEncoder()
categorical_columns = ['Gender', 'Blood Type', 'Medical Condition', 
                      'Insurance Provider', 'Medication', 'Test Results']

for column in categorical_columns:
    df[column] = le.fit_transform(df[column])

# Scale numerical features
scaler = StandardScaler()
df['Age'] = scaler.fit_transform(df[['Age']])
df['Billing Amount'] = scaler.fit_transform(df[['Billing Amount']])

# Prepare features and target
X = df.drop('Admission Type', axis=1)
y = le.fit_transform(df['Admission Type'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Initial Model Training and Evaluation
print("\n1. Initial Model Evaluation:")
print("----------------------------")
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Cross-validation (reduced to 3 folds)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=3, n_jobs=-1)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Initial predictions
y_pred = rf_model.predict(X_test)
print("\nInitial Model Performance:")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 2. Model Tuning
print("\n2. Model Tuning:")
print("---------------")
# Define parameter grid (reduced number of values)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Perform grid search (reduced to 3 folds)
grid_search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), 
                         param_grid, 
                         cv=3, 
                         scoring='accuracy',
                         n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print tuning results
print("\nBest parameters:", grid_search.best_params_)
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# 3. Evaluate Tuned Model
print("\n3. Tuned Model Evaluation:")
print("-------------------------")
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

print(f"Tuned Model Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")
print("\nTuned Model Classification Report:")
print(classification_report(y_test, y_pred_tuned))

# 4. Visualizations
print("\n4. Generating Visualizations...")
print("----------------------------")

# Create figure for feature importance comparison
plt.figure(figsize=(15, 5))

# Initial Model Feature Importance
plt.subplot(1, 2, 1)
feature_importance_initial = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance_initial = feature_importance_initial.sort_values('importance', ascending=False)

sns.barplot(x='importance', y='feature', data=feature_importance_initial)
plt.title('Feature Importance (Initial Model)')
plt.xlabel('Importance')

# Tuned Model Feature Importance
plt.subplot(1, 2, 2)
feature_importance_tuned = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
})
feature_importance_tuned = feature_importance_tuned.sort_values('importance', ascending=False)

sns.barplot(x='importance', y='feature', data=feature_importance_tuned)
plt.title('Feature Importance (Tuned Model)')
plt.xlabel('Importance')

plt.tight_layout()
plt.show()

# Create figure for confusion matrix comparison
plt.figure(figsize=(15, 5))

# Initial Model Confusion Matrix
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Initial Model)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Tuned Model Confusion Matrix
plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_tuned), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Tuned Model)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# Performance comparison
print("\n5. Performance Comparison:")
print("------------------------")
print(f"Initial Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Tuned Model Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")