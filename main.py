import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# ----------------------------------------------
# 1. Identifying and Sourcing Relevant Dataset
# ----------------------------------------------
# Sample dataset (can be replaced with actual)
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/StudentPerformance.csv')

# ----------------------------------------------
# 2. Cleaning and Handling Missing Values
# ----------------------------------------------
print("\nMissing values before imputation:\n", df.isnull().sum())
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
print("\nMissing values after imputation:\n", df_imputed.isnull().sum())

# ----------------------------------------------
# 3. Feature Selection and Engineering
# ----------------------------------------------
# Encode categorical variables
cat_cols = df_imputed.select_dtypes(include='object').columns
encoder = LabelEncoder()
for col in cat_cols:
    df_imputed[col] = encoder.fit_transform(df_imputed[col])

# Example feature engineering: GPA average
df_imputed['avg_grade'] = df_imputed[['G1', 'G2', 'G3']].mean(axis=1)

# ----------------------------------------------
# 4. Ensuring Data Integrity and Consistency
# ----------------------------------------------
# Check duplicates and inconsistent values
df_cleaned = df_imputed.drop_duplicates()
assert df_cleaned.isnull().sum().sum() == 0  # No missing
print("\nDataset cleaned. Shape:", df_cleaned.shape)

# ----------------------------------------------
# 5. Summary Statistics and Insights
# ----------------------------------------------
print("\nSummary Statistics:\n", df_cleaned.describe())
print("\nCorrelation Matrix:\n", df_cleaned.corr()['G3'].sort_values(ascending=False))

# ----------------------------------------------
# 6. Identifying Patterns, Trends, and Anomalies
# ----------------------------------------------
sns.heatmap(df_cleaned.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix Heatmap")
plt.show()

sns.boxplot(x='failures', y='avg_grade', data=df_cleaned)
plt.title("Failures vs Average Grade")
plt.show()

# ----------------------------------------------
# 7. Handling Outliers and Data Transformations
# ----------------------------------------------
# Outlier detection (example: grades)
Q1 = df_cleaned['avg_grade'].quantile(0.25)
Q3 = df_cleaned['avg_grade'].quantile(0.75)
IQR = Q3 - Q1
outliers = df_cleaned[(df_cleaned['avg_grade'] < (Q1 - 1.5 * IQR)) | (df_cleaned['avg_grade'] > (Q3 + 1.5 * IQR))]
print("\nNumber of Outliers in avg_grade:", len(outliers))

# Scaling for PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_cleaned.drop(columns=['G3']))

# ----------------------------------------------
# 8. Initial Visual Representation of Key Findings
# ----------------------------------------------
# PCA Visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue=df_cleaned['G3'], palette='cool', data=df_pca)
plt.title('PCA - Student Dropout Dataset')
plt.show()
