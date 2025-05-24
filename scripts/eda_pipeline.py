import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import os

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = os.path.join('..', 'data', 'student_data.csv')  # Adjust path if needed
OUTPUT_DIR = os.path.join('..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Step 1: Load the dataset
# -----------------------------
print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Shape of dataset:", df.shape)

# -----------------------------
# Step 2: Handle missing values
# -----------------------------
print("[INFO] Handling missing values...")
imputer = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# -----------------------------
# Step 3: Encode categorical features
# -----------------------------
print("[INFO] Encoding categorical variables...")
label_enc = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_enc.fit_transform(df[col])

# -----------------------------
# Step 4: Feature engineering
# -----------------------------
print("[INFO] Feature engineering...")
if {'G1', 'G2', 'G3'}.issubset(df.columns):
    df['avg_grade'] = df[['G1', 'G2', 'G3']].astype(float).mean(axis=1)

# -----------------------------
# Step 5: Data consistency checks
# -----------------------------
print("[INFO] Removing duplicate rows...")
df.drop_duplicates(inplace=True)

# -----------------------------
# Step 6: Summary statistics
# -----------------------------
print("\n[INFO] Summary Statistics:")
print(df.describe())

# -----------------------------
# Step 7: Correlation matrix
# -----------------------------
print("[INFO] Creating correlation matrix heatmap...")
corr = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))
plt.close()

# -----------------------------
# Step 8: Detect outliers (on avg_grade)
# -----------------------------
print("[INFO] Detecting outliers in avg_grade...")
if 'avg_grade' in df.columns:
    Q1 = df['avg_grade'].quantile(0.25)
    Q3 = df['avg_grade'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['avg_grade'] < Q1 - 1.5 * IQR) | (df['avg_grade'] > Q3 + 1.5 * IQR)]
    print("Outliers detected:", outliers.shape[0])

# -----------------------------
# Step 9: PCA for visualization
# -----------------------------
print("[INFO] Performing PCA for visualization...")
features = df.drop(columns=['G3']) if 'G3' in df.columns else df.copy()
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

pca = PCA(n_components=2)
components = pca.fit_transform(scaled)
df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue=df['G3'] if 'G3' in df.columns else None, palette='cool', data=df_pca)
plt.title("PCA - Student Dropout Risk Dataset")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_scatter.png'))
plt.close()

print("[INFO] EDA Pipeline completed successfully.")
