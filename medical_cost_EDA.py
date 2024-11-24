import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('insurance.csv')

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

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Histograms for Age, BMI, and Charges
df[['age', 'bmi', 'charges']].hist(figsize=(15, 5), bins=20, layout=(1, 3))
plt.suptitle("Distribution of Age, BMI, and Charges")
plt.show()

# Boxplot for Age, BMI, and Charges
plt.figure(figsize=(12, 5))
sns.boxplot(data=df[['age', 'bmi', 'charges']])
plt.title("Boxplot of Age, BMI, and Charges")
plt.show()

# Bar plot for Children
plt.figure(figsize=(8, 5))
sns.countplot(x='children', data=df)
plt.title("Count of Children")
plt.show()

# Bar plots for Categorical Variables: Sex, Smoker, and Region
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.countplot(ax=axes[0], x='sex', data=df)
axes[0].set_title("Distribution of Sex")

sns.countplot(ax=axes[1], x='smoker', data=df)
axes[1].set_title("Distribution of Smoker")

sns.countplot(ax=axes[2], x='region', data=df)
axes[2].set_title("Distribution of Region")

plt.show()

# Scatter plots to visualize relationships between variables
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(ax=axes[0], x='age', y='charges', data=df)
axes[0].set_title("Age vs Charges")

sns.scatterplot(ax=axes[1], x='bmi', y='charges', data=df)
axes[1].set_title("BMI vs Charges")

plt.show()

# Correlation heatmap to explore relationships between numerical features
plt.figure(figsize=(10, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 2. Data Preprocessing

# One-hot encoding categorical features
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Feature scaling for numerical features
scaler = StandardScaler()
df[['age', 'bmi', 'children']] = scaler.fit_transform(df[['age', 'bmi', 'children']])

# Define features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display shapes of train and test sets
print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

