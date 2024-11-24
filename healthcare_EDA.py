import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv('healthcare_dataset.csv')

# 1. Exploratory Data Analysis (EDA)

# Dataset Overview
print("Dataset Overview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nCheck for Missing Values:")
print(df.isnull().sum())

print("\nCheck for Duplicates:")
print(df.duplicated().sum())

# Summary statistics for numerical variables (Age and Billing Amount)
print("\nSummary Statistics for Numerical Variables:")
print(df[['Age', 'Billing Amount']].describe())

# Distribution plots for numerical variables
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='Age', bins=30)
plt.title('Distribution of Patient Ages')

plt.subplot(1, 2, 2)
sns.histplot(data=df, x='Billing Amount', bins=30)
plt.title('Distribution of Billing Amounts')
plt.tight_layout()
plt.show()

# Boxplots for numerical variables
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, y='Age')
plt.title('Age Distribution')

plt.subplot(1, 2, 2)
sns.boxplot(data=df, y='Billing Amount')
plt.title('Billing Amount Distribution')
plt.tight_layout()
plt.show()

# Distribution of categorical variables
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.countplot(data=df, x='Admission Type')
plt.title('Distribution of Admission Types')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
sns.countplot(data=df, x='Gender')
plt.title('Gender Distribution')

plt.subplot(2, 2, 3)
sns.countplot(data=df, x='Blood Type')
plt.title('Blood Type Distribution')
plt.xticks(rotation=45)

plt.subplot(2, 2, 4)
sns.countplot(data=df, x='Medical Condition', order=df['Medical Condition'].value_counts().index[:10])
plt.title('Top 10 Medical Conditions')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Relationship between Age and Admission Type
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Admission Type', y='Age')
plt.title('Age Distribution by Admission Type')
plt.show()

# Relationship between Billing Amount and Admission Type
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Admission Type', y='Billing Amount')
plt.title('Billing Amount Distribution by Admission Type')
plt.show()

# Medical Condition vs Admission Type
plt.figure(figsize=(12, 6))
condition_admission = pd.crosstab(df['Medical Condition'], df['Admission Type'])
condition_admission_percent = condition_admission.div(condition_admission.sum(axis=1), axis=0)
sns.heatmap(condition_admission_percent, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Medical Condition vs Admission Type (Normalized)')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Test Results vs Admission Type
plt.figure(figsize=(10, 6))
test_admission = pd.crosstab(df['Test Results'], df['Admission Type'])
sns.heatmap(test_admission, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Test Results vs Admission Type')
plt.tight_layout()
plt.show()

# 2. Data Preprocessing

# Convert categorical variables to numeric
le = LabelEncoder()
categorical_columns = ['Gender', 'Blood Type', 'Medical Condition', 
                      'Insurance Provider', 'Medication', 'Test Results']

for column in categorical_columns:
    df[column] = le.fit_transform(df[column])

# Scale numerical features
scaler = StandardScaler()
df[['Age', 'Billing Amount']] = scaler.fit_transform(df[['Age', 'Billing Amount']])

# Prepare features and target
X = df.drop(['Admission Type', 'Name', 'Doctor', 'Room Number', 
             'Date of Admission', 'Discharge Date', 'Hospital'], axis=1)
y = le.fit_transform(df['Admission Type'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display shapes of train and test sets
print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

