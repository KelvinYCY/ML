import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("Loading and preparing data...")
df = pd.read_csv('healthcare_dataset.csv')

# Use fewer columns - only those most relevant for prediction
columns_to_keep = ['Age', 'Gender', 'Medical Condition', 'Insurance Provider', 
                   'Billing Amount', 'Test Results', 'Admission Type']
df = df[columns_to_keep]

# Convert categorical variables to numerical
le = LabelEncoder()
categorical_columns = ['Gender', 'Medical Condition', 'Insurance Provider', 'Test Results']

for column in categorical_columns:
    df[column] = le.fit_transform(df[column])

# Scale numerical features
scaler = StandardScaler()
df[['Age', 'Billing Amount']] = scaler.fit_transform(df[['Age', 'Billing Amount']])

# Prepare features and target
X = df.drop('Admission Type', axis=1)
y = le.fit_transform(df['Admission Type'])

# Use a smaller subset for initial testing (20% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Initial Model with basic parameters
print("\n1. Initial Model Evaluation:")
print("----------------------------")
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Reduced cross-validation folds
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=3)  # Reduced from 5 to 3
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Initial predictions
y_pred = svm_model.predict(X_test)
print("\nInitial Model Performance:")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 2. Model Tuning with reduced parameter grid
print("\n2. Model Tuning:")
print("---------------")
# Simplified parameter grid
param_grid = {
    'C': [1, 10],           # Reduced from 4 to 2 values
    'gamma': ['scale', 0.1], # Reduced from 4 to 2 values
    'kernel': ['rbf']       # Using only the best kernel
}

# Perform grid search with fewer cross-validation folds
grid_search = GridSearchCV(SVC(random_state=42),
                         param_grid,
                         cv=3,  # Reduced from 5 to 3
                         scoring='accuracy',
                         n_jobs=2)  # Limit CPU cores used
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

# Plot confusion matrices
plt.figure(figsize=(12, 5))

# Initial model confusion matrix
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Initial Model)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Tuned model confusion matrix
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