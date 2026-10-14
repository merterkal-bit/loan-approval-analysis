import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1 & 2. Load and Clean Data
df = pd.read_csv("loan_approval_dataset.csv", sep=";")
df.columns = df.columns.str.strip()

# Remove whitespace from text columns
for col in df.select_dtypes("object").columns:
    df[col] = df[col].str.strip()

# Requirement 2: Print the first 20 rows
print("--- FIRST 20 ROWS ---")
print(df.head(20))

# Helper for Requirement 3: Print all features to aid manual table creation
print("\n--- ALL FEATURES IN DATASET ---")
print(df.columns.tolist())

# Requirement 4a: Convert Target Variable to Numeric & Compute Correlation
df["Loan_Status_Numeric"] = df["loan_status"].map({
    "Rejected": 0,
    "Approved": 1
})

df["education_encoded"] = df["education"].map({"Not Graduate": 0, "Graduate": 1})
df["self_employed_encoded"] = df["self_employed"].map({"No": 0, "Yes": 1})

# Calculate correlation matrix for numerical features
corrmatrix = df.select_dtypes(include=np.number).corr()

# Requirement 4b: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corrmatrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Numerical Features")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", bbox_inches="tight", dpi=300)
plt.show()

# Requirement 5: Categorical Feature Analysis (Education vs Loan Status)
education_pct = pd.crosstab(
    df["education"],
    df["loan_status"],
    normalize="index"
) * 100

ax = education_pct.plot(kind="bar", figsize=(8, 6))
plt.ylabel("Percentage (%)")
plt.title("Loan Status Distribution by Education Level")
plt.xticks(rotation=0)
plt.legend(title="Loan Status", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("education_vs_loan_status.png", bbox_inches="tight", dpi=300)
plt.show()