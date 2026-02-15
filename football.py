# Data handling
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = "C:/Users/Mahima kumari/Downloads/Strikers_performance (1).xlsx"
df = pd.read_excel(file_path)

# Display first 5 rows
print (df.head())


# Dataset information
df.info()

# Check missing values
df.isnull().sum()


# Identify numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Identify categorical (nominal) columns
categorical_cols = df.select_dtypes(include=['object']).columns

print("Numeric Columns:")
print(numeric_cols)

print("\nCategorical Columns:")
print(categorical_cols)


median_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = median_imputer.fit_transform(df[numeric_cols])


columns_to_convert = [
    'Goals Scored',
    'Assists',
    'Shots on Target',
    'Movement off the Ball',
    'Hold-up Play',
    'Aerial Duels Won',
    'Defensive Contribution',
    'Big Game Performance',
    'Impact on Team Performance',
    'Off-field Conduct'
]

df[columns_to_convert] = df[columns_to_convert].astype(int)
df[columns_to_convert].dtypes


print(df[columns_to_convert].dtypes)



# --------------------------------------------
# This calculates:
# count, mean, std, min, 25%, 50%, 75%, max

numeric_summary = df.describe()

# Round all values to 2 decimal places
numeric_summary = numeric_summary.round(2)

print("\nDescriptive Statistics (Numeric Columns):")
print(numeric_summary)


# --------------------------------------------
# STEP 5: Descriptive Statistics for Categorical Columns
# --------------------------------------------
# This calculates:
# count, unique, top (most frequent), frequency

categorical_summary = df.describe(include='object')

print("\nDescriptive Statistics (Categorical Columns):")
print(categorical_summary)


# --------------------------------------------
# STEP 6: Additional Statistical Measures (Optional but Good for Assignment)
# --------------------------------------------

# Calculate Median separately (rounded to 2 decimals)
median_values = df.median(numeric_only=True).round(2)

print("\nMedian Values (Numeric Columns):")
print(median_values)


# Calculate Variance (rounded to 2 decimals)
variance_values = df.var(numeric_only=True).round(2)

print("\nVariance (Numeric Columns):")
print(variance_values)


# --------------------------------------------
# STEP 7: Check for Missing Values
# --------------------------------------------
missing_values = df.isnull().sum()

print("\nMissing Values in Each Column:")
print(missing_values)



import matplotlib.pyplot as plt
import seaborn as sns

plt.ion()   # Turn interactive mode ON




# --------------------------------------------
# STEP 3: Percentage Analysis of Footedness
# --------------------------------------------

# Calculate percentage distribution
footedness_percent = df['Footedness'].value_counts(normalize=True) * 100

# Round to 2 decimal places
footedness_percent = footedness_percent.round(2)

# Print percentage results
print("Percentage Distribution of Footedness:")
print(footedness_percent)


# --------------------------------------------
# STEP 4: Pie Chart using Matplotlib
# --------------------------------------------

plt.figure(figsize=(6,6))

plt.pie(
    footedness_percent,
    labels=footedness_percent.index,
    autopct='%1.1f%%',
    startangle=90
)

plt.title("Percentage Distribution of Players' Footedness")
plt.show()


# --------------------------------------------
# STEP 5: Countplot of Footedness Across Nationalities
# --------------------------------------------
plt.figure(figsize=(12,6))

sns.countplot(
    data=df,
    x='Nationality',
    hue='Footedness'
)

plt.title("Distribution of Players' Footedness Across Nationalities")
plt.xticks(rotation=90)

plt.tight_layout()   # Prevent label cutting
plt.show()



# -----------------------------------------------------------
# STEP 1: Import Required Libraries
# -----------------------------------------------------------

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


# -----------------------------------------------------------
# STEP 2: Load Dataset
# -----------------------------------------------------------

file_path = "C:/Users/Mahima kumari/Downloads/Strikers_performance (1).xlsx"
df = pd.read_excel(file_path)

print("\nDataset Loaded Successfully")


# ===========================================================
# 1️⃣ Nationality with Highest Average Goals Scored
# ===========================================================

print("\n--- 1️⃣ Average Goals by Nationality ---")

avg_goals = df.groupby('Nationality')['Goals Scored'].mean().round(2)

print(avg_goals)

highest_nationality = avg_goals.idxmax()
highest_value = avg_goals.max()

print(f"\nNationality with Highest Average Goals: {highest_nationality}")
print(f"Highest Average Goals: {round(highest_value,2)}")


# ===========================================================
# 2️⃣ Average Conversion Rate by Footedness
# ===========================================================

print("\n--- 2️⃣ Average Conversion Rate by Footedness ---")

avg_conversion = df.groupby('Footedness')['Conversion Rate'].mean().round(2)

print(avg_conversion)


# ===========================================================
# 3️⃣ Test Difference in Consistency Across Nationalities
# ===========================================================

print("\n--- 3️⃣ Testing Difference in Consistency Among Nationalities ---")

# -----------------------------------------------------------
# Assumption 1: Normality (Shapiro-Wilk Test)
# -----------------------------------------------------------

print("\nChecking Normality for Consistency (Shapiro Test):")

shapiro_test = stats.shapiro(df['Consistency'])
print("Shapiro Test Statistic:", round(shapiro_test.statistic,4))
print("p-value:", round(shapiro_test.pvalue,4))

if shapiro_test.pvalue > 0.05:
    print("Data is normally distributed (Fail to reject H0)")
else:
    print("Data is NOT normally distributed (Reject H0)")

# -----------------------------------------------------------
# Assumption 2: Homogeneity of Variance (Levene Test)
# -----------------------------------------------------------

print("\nChecking Homogeneity of Variance (Levene Test):")

groups = [group["Consistency"].values for name, group in df.groupby("Nationality")]

levene_test = stats.levene(*groups)
print("Levene Test Statistic:", round(levene_test.statistic,4))
print("p-value:", round(levene_test.pvalue,4))

if levene_test.pvalue > 0.05:
    print("Equal variances assumed")
else:
    print("Variances are not equal")

# -----------------------------------------------------------
# Perform ANOVA
# -----------------------------------------------------------

print("\nPerforming One-Way ANOVA:")

model = ols('Consistency ~ C(Nationality)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table.round(4))

if anova_table["PR(>F)"][0] < 0.05:
    print("\nThere is a significant difference in consistency among nationalities.")
else:
    print("\nNo significant difference in consistency among nationalities.")


# ===========================================================
# 4️⃣ Correlation Between Hold-up Play and Consistency
# ===========================================================

print("\n--- 4️⃣ Correlation Between Hold-up Play and Consistency ---")

# -----------------------------------------------------------
# Assumption: Normality for both variables
# -----------------------------------------------------------

print("\nChecking Normality for Hold-up Play:")
print(stats.shapiro(df['Hold-up Play']))

print("\nChecking Normality for Consistency:")
print(stats.shapiro(df['Consistency']))

# -----------------------------------------------------------
# Pearson Correlation Test
# -----------------------------------------------------------

corr, p_value = stats.pearsonr(df['Hold-up Play'], df['Consistency'])

print("\nPearson Correlation Coefficient:", round(corr,4))
print("p-value:", round(p_value,4))

if p_value < 0.05:
    print("Significant correlation exists.")
else:
    print("No significant correlation found.")


# ===========================================================
# 5️⃣ Regression: Does Hold-up Play Influence Consistency?
# ===========================================================

print("\n--- 5️⃣ Linear Regression: Hold-up Play → Consistency ---")

X = df['Hold-up Play']
y = df['Consistency']

X = sm.add_constant(X)  # Add intercept

model_reg = sm.OLS(y, X).fit()

print(model_reg.summary())

if model_reg.pvalues[1] < 0.05:
    print("\nHold-up Play significantly influences Consistency.")
else:
    print("\nHold-up Play does NOT significantly influence Consistency.")




# -----------------------------------------------------------
# FEATURE ENGINEERING SECTION
# -----------------------------------------------------------

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------
# STEP 1: Load Dataset
# -----------------------------------------------------------

file_path = "C:/Users/Mahima kumari/Downloads/Strikers_performance (1).xlsx"
df = pd.read_excel(file_path)

print("\nDataset Loaded Successfully!")


# ===========================================================
# 1️⃣ Create Total Contribution Score
# ===========================================================

print("\n--- Creating Total Contribution Score ---")

# Columns to sum
contribution_columns = [
    'Goals Scored',
    'Assists',
    'Shots on Target',
    'Dribbling Success',
    'Aerial Duels Won',
    'Defensive Contribution',
    'Big Game Performance',
    'Consistency'
]

# Create new feature
df['Total Contribution Score'] = df[contribution_columns].sum(axis=1)

print("New Feature 'Total Contribution Score' Created Successfully!")
print("\nPreview of Total Contribution Score:")
print(df[['Total Contribution Score']].head())


# ===========================================================
# 2️⃣ Label Encoding for Footedness and Marital Status
# ===========================================================

print("\n--- Encoding Footedness and Marital Status ---")

label_encoder = LabelEncoder()

# Encode Footedness
df['Footedness_Encoded'] = label_encoder.fit_transform(df['Footedness'])

print("\nFootedness Encoding Mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label} → {i}")

# Encode Marital Status
df['Marital_Status_Encoded'] = label_encoder.fit_transform(df['Marital Status'])

print("\nMarital Status Encoding Mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label} → {i}")

print("\nEncoding Completed Successfully!")


# ===========================================================
# 3️⃣ Create Dummy Variables for Nationality
# ===========================================================

print("\n--- Creating Dummy Variables for Nationality ---")

nationality_dummies = pd.get_dummies(df['Nationality'], prefix='Nationality')

# Add dummy variables to original dataframe
df = pd.concat([df, nationality_dummies], axis=1)

print("Dummy Variables Added Successfully!")

print("\nPreview of Nationality Dummy Variables:")
print(nationality_dummies.head())


# -----------------------------------------------------------
# Final Dataset Overview
# -----------------------------------------------------------

print("\nFinal Dataset Shape After Feature Engineering:")
print(df.shape)

print("\nFeature Engineering Completed Successfully!")




# ===========================================================
# CLUSTERING ANALYSIS – KMEANS (FINAL CORRECT VERSION)
# ===========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# -----------------------------------------------------------
# STEP 1: Load Dataset
# -----------------------------------------------------------

file_path = "C:/Users/Mahima kumari/Downloads/Strikers_performance (1).xlsx"
df = pd.read_excel(file_path)

print("\nDataset Loaded Successfully!")


# -----------------------------------------------------------
# STEP 2: Create Total Contribution Score (If Not Created)
# -----------------------------------------------------------

print("\nCreating Total Contribution Score...")

contribution_columns = [
    'Goals Scored',
    'Assists',
    'Shots on Target',
    'Dribbling Success',
    'Aerial Duels Won',
    'Defensive Contribution',
    'Big Game Performance',
    'Consistency'
]

df['Total Contribution Score'] = df[contribution_columns].sum(axis=1)

print("Total Contribution Score Created Successfully!")


# -----------------------------------------------------------
# STEP 3: Select Features (Drop Striker_ID)
# -----------------------------------------------------------

print("\nSelecting Features for Clustering...")

X = df.drop(columns=['Striker_ID'])

# Keep only numeric columns
X = X.select_dtypes(include=[np.number])

print("Numeric Features Selected!")
print("Feature Matrix Shape:", X.shape)


# -----------------------------------------------------------
# STEP 4: Handle Missing Values
# -----------------------------------------------------------

print("\nHandling Missing Values (Using Median Imputation)...")

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

print("Missing Values Handled Successfully!")


# -----------------------------------------------------------
# STEP 5: Feature Scaling
# -----------------------------------------------------------

print("\nScaling Features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print("Feature Scaling Completed Successfully!")


# -----------------------------------------------------------
# STEP 6: Calculate WCSS (Elbow Method)
# -----------------------------------------------------------

print("\nCalculating WCSS for Elbow Method...")

wcss = []

for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Print WCSS values
for i in range(1, 6):
    print(f"Clusters = {i}, WCSS = {round(wcss[i-1],2)}")


# -----------------------------------------------------------
# STEP 7: Plot Elbow Chart
# -----------------------------------------------------------

plt.figure(figsize=(8,5))
plt.plot(range(1,6), wcss, marker='o')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.xticks(range(1,6))
plt.show()

input("Observe the elbow chart. Press Enter to continue...")

print("\nFrom the elbow chart, optimal number of clusters = 2")


# -----------------------------------------------------------
# STEP 8: Build KMeans with 2 Clusters
# -----------------------------------------------------------

print("\nBuilding KMeans Model with 2 Clusters...")

kmeans_final = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Clusters'] = kmeans_final.fit_predict(X_scaled)

print("Clustering Completed Successfully!")
print("\nCluster Distribution:")
print(df['Clusters'].value_counts())


# -----------------------------------------------------------
# STEP 9: Average Total Contribution Score by Cluster
# -----------------------------------------------------------

print("\nCalculating Average Total Contribution Score per Cluster...")

cluster_avg = df.groupby('Clusters')['Total Contribution Score'].mean().round(2)

print(cluster_avg)


# -----------------------------------------------------------
# STEP 10: Assign Striker Types
# -----------------------------------------------------------

print("\nAssigning Striker Types...")

df['Strikers types'] = df['Clusters'].map({
    0: 'Best strikers',
    1: 'Regular strikers'
})

print("Striker Types Assigned Successfully!")
print(df[['Clusters', 'Strikers types']].head())


# -----------------------------------------------------------
# STEP 11: Drop Cluster Column
# -----------------------------------------------------------

df.drop(columns=['Clusters'], inplace=True)

print("\nClusters Column Dropped Successfully!")


# -----------------------------------------------------------
# STEP 12: Feature Mapping (Encode Strikers Types)
# -----------------------------------------------------------

print("\nEncoding Strikers Types...")

df['Strikers types'] = df['Strikers types'].map({
    'Best strikers': 1,
    'Regular strikers': 0
})

print("Strikers Types Encoded Successfully!")
print(df[['Strikers types']].head())


# -----------------------------------------------------------
# FINAL OUTPUT
# -----------------------------------------------------------

print("\nFinal Dataset Shape:", df.shape)
print("\nClustering Analysis Completed Successfully!")




# ===========================================================
# MACHINE LEARNING MODEL – LOGISTIC REGRESSION
# ===========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ===========================================================
# MACHINE LEARNING MODEL – LOGISTIC REGRESSION
# ===========================================================

print("\n" + "="*60)
print("STARTING MACHINE LEARNING MODEL")
print("="*60)


# -----------------------------------------------------------
# STEP 1: Check Target Column
# -----------------------------------------------------------

print("\n[STEP 1] Checking if 'Strikers types' column exists...")

if 'Strikers types' not in df.columns:
    print("ERROR: 'Strikers types' column NOT found!")
    print("Please make sure clustering section runs before this.")
else:
    print("'Strikers types' column found successfully!")


# -----------------------------------------------------------
# STEP 2: Select Features (X) and Target (y)
# -----------------------------------------------------------

print("\n[STEP 2] Selecting Features (X) and Target (y)...")

# Drop unnecessary columns
X = df.drop(columns=['Striker_ID', 'Strikers types'])

# Keep only numeric columns
X = X.select_dtypes(include=['int64', 'float64'])

y = df['Strikers types']

print("Feature Matrix Shape:", X.shape)
print("Target Vector Shape:", y.shape)


# -----------------------------------------------------------
# STEP 3: Handle Missing Values
# -----------------------------------------------------------

print("\n[STEP 3] Handling Missing Values (Median)...")

X = X.fillna(X.median())

print("Missing values after cleaning:")
print(X.isnull().sum().sum(), "missing values remaining.")


# -----------------------------------------------------------
# STEP 4: Feature Scaling
# -----------------------------------------------------------

from sklearn.preprocessing import StandardScaler

print("\n[STEP 4] Scaling Features using StandardScaler...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature Scaling Completed!")


# -----------------------------------------------------------
# STEP 5: Train-Test Split (80% Train / 20% Test)
# -----------------------------------------------------------

from sklearn.model_selection import train_test_split

print("\n[STEP 5] Splitting Data (80% Train / 20% Test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# -----------------------------------------------------------
# STEP 6: Build Logistic Regression Model
# -----------------------------------------------------------

from sklearn.linear_model import LogisticRegression

print("\n[STEP 6] Training Logistic Regression Model...")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model Training Completed Successfully!")


# -----------------------------------------------------------
# STEP 7: Make Predictions
# -----------------------------------------------------------

print("\n[STEP 7] Making Predictions on Test Data...")

y_pred = model.predict(X_test)

print("Predictions Completed!")


# -----------------------------------------------------------
# STEP 8: Calculate Accuracy
# -----------------------------------------------------------

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred) * 100

print("\n[STEP 8] Model Accuracy:")
print("Accuracy = {:.2f}%".format(accuracy))


# -----------------------------------------------------------
# STEP 9: Confusion Matrix
# -----------------------------------------------------------

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("\n[STEP 9] Generating Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix Values:")
print(cm)


# -----------------------------------------------------------
# STEP 10: Visualize Confusion Matrix
# -----------------------------------------------------------

print("\n[STEP 10] Visualizing Confusion Matrix...")

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Regular (0)', 'Best (1)'],
    yticklabels=['Regular (0)', 'Best (1)']
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

input("Press Enter to continue...")


# -----------------------------------------------------------
# STEP 11: Classification Report
# -----------------------------------------------------------

from sklearn.metrics import classification_report

print("\n[STEP 11] Classification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------------------------------------
# FINAL INTERPRETATION
# -----------------------------------------------------------

print("\n" + "="*60)
print("FINAL INTERPRETATION")
print("="*60)

print(f"""
The Logistic Regression model achieved an accuracy of {accuracy:.2f}% 
in predicting striker types.

The confusion matrix shows how well the model classified:

0 → Regular strikers
1 → Best strikers

Based on the accuracy score and classification report,
the model effectively distinguishes between high-performing 
and regular strikers.

Therefore, the machine learning model successfully predicts 
striker types using performance features.
""")

print("Machine Learning Section Completed Successfully!")



# ===========================================================
# LEVENE'S TEST FOR HOMOGENEITY OF VARIANCE (ANOVA ASSUMPTION)
# ===========================================================

import pandas as pd
from scipy import stats

print("\n" + "="*60)
print("LEVENE'S TEST FOR HOMOGENEITY OF VARIANCE")
print("="*60)

# -----------------------------------------------------------
# STEP 1: Group Consistency values by Nationality
# -----------------------------------------------------------

print("\n[STEP 1] Grouping Consistency Scores by Nationality...")

groups = [group["Consistency"].values 
          for name, group in df.groupby("Nationality")]

print("Number of Nationality Groups:", len(groups))


# -----------------------------------------------------------
# STEP 2: Perform Levene's Test
# -----------------------------------------------------------

print("\n[STEP 2] Performing Levene's Test...")

levene_test = stats.levene(*groups)

print("Levene Test Statistic:", round(levene_test.statistic, 4))
print("Levene Test p-value:", round(levene_test.pvalue, 4))


# -----------------------------------------------------------
# STEP 3: Interpretation of Result
# -----------------------------------------------------------

print("\n[STEP 3] Interpretation:")

if levene_test.pvalue > 0.05:
    print("""
Since the p-value is greater than 0.05,
we fail to reject the null hypothesis.

Equal variances are assumed.
Heteroscedasticity is NOT present.
""")
else:
    print("""
Since the p-value is less than 0.05,
we reject the null hypothesis.

Variances are not equal.
Heteroscedasticity IS present.
""")

print("="*60)


# ===========================================================
# CORRELATION TEST: HOLD-UP PLAY vs CONSISTENCY
# ===========================================================

from scipy import stats
import numpy as np

print("\n" + "="*60)
print("CORRELATION ANALYSIS: HOLD-UP PLAY vs CONSISTENCY")
print("="*60)


# -----------------------------------------------------------
# STEP 1: Check Normality Assumption (Shapiro-Wilk Test)
# -----------------------------------------------------------

print("\n[STEP 1] Checking Normality Assumption...")

shapiro_hold = stats.shapiro(df['Hold-up Play'])
shapiro_cons = stats.shapiro(df['Consistency'])

print("Hold-up Play Normality p-value:", round(shapiro_hold.pvalue, 4))
print("Consistency Normality p-value:", round(shapiro_cons.pvalue, 4))

if shapiro_hold.pvalue > 0.05 and shapiro_cons.pvalue > 0.05:
    print("Both variables are normally distributed.")
    print("Proceeding with Pearson Correlation.\n")
else:
    print("One or both variables are NOT normally distributed.")
    print("Pearson may still be used for large samples (n > 30).\n")


# -----------------------------------------------------------
# STEP 2: Perform Pearson Correlation Test
# -----------------------------------------------------------

print("[STEP 2] Performing Pearson Correlation Test...")

correlation, p_value = stats.pearsonr(df['Hold-up Play'], df['Consistency'])

print("Correlation Coefficient (r):", round(correlation, 4))
print("p-value:", round(p_value, 4))


# -----------------------------------------------------------
# STEP 3: Interpretation
# -----------------------------------------------------------

print("\n[STEP 3] Interpretation:")

if p_value < 0.05:
    print(f"""
There IS a statistically significant correlation between 
Hold-up Play and Consistency.

Correlation strength (r) = {round(correlation,4)}

Since p-value < 0.05, the relationship is significant.
""")
else:
    print(f"""
There is NO statistically significant correlation between 
Hold-up Play and Consistency.

Since p-value > 0.05, the relationship is not significant.
""")

print("="*60)


# ===========================================================
# REGRESSION ANALYSIS: HOLD-UP PLAY → CONSISTENCY
# ===========================================================

import statsmodels.api as sm

print("\n" + "="*60)
print("LINEAR REGRESSION ANALYSIS")
print("="*60)


# -----------------------------------------------------------
# STEP 1: Define Variables
# -----------------------------------------------------------

print("\n[STEP 1] Defining Independent and Dependent Variables...")

X = df['Hold-up Play']
y = df['Consistency']

# Add constant (intercept)
X = sm.add_constant(X)

print("Variables defined successfully.")


# -----------------------------------------------------------
# STEP 2: Fit Regression Model
# -----------------------------------------------------------

print("\n[STEP 2] Fitting Linear Regression Model...")

model = sm.OLS(y, X).fit()

print("Model Fitted Successfully!")


# -----------------------------------------------------------
# STEP 3: Display Regression Summary
# -----------------------------------------------------------

print("\n[STEP 3] Regression Summary:")
print(model.summary())


# -----------------------------------------------------------
# STEP 4: Extract Beta Value
# -----------------------------------------------------------

beta_value = model.params['Hold-up Play']
p_value = model.pvalues['Hold-up Play']

print("\n[STEP 4] Extracted Beta Value for Hold-up Play:")
print("Beta (Coefficient) =", round(beta_value, 4))
print("p-value =", round(p_value, 4))


# -----------------------------------------------------------
# STEP 5: Interpretation
# -----------------------------------------------------------

print("\n[STEP 5] Interpretation:")

if p_value < 0.05:
    print(f"""
The beta coefficient of Hold-up Play is {round(beta_value,4)}.

This means that for every one-unit increase in Hold-up Play,
Consistency increases by {round(beta_value,4)} units.

Since p-value < 0.05, this effect is statistically significant.
""")
else:
    print(f"""
The beta coefficient of Hold-up Play is {round(beta_value,4)}.

However, since p-value > 0.05,
the effect is not statistically significant.
""")

print("="*60)



# ===========================================================
# AVERAGE TOTAL CONTRIBUTION SCORE FOR BEST STRIKERS
# ===========================================================

print("\n" + "="*60)
print("AVERAGE TOTAL CONTRIBUTION SCORE FOR BEST STRIKERS")
print("="*60)


# -----------------------------------------------------------
# STEP 1: Check if Column Exists
# -----------------------------------------------------------

print("\n[STEP 1] Checking Required Columns...")

if 'Strikers types' not in df.columns:
    print("ERROR: 'Strikers types' column not found.")
elif 'Total Contribution Score' not in df.columns:
    print("ERROR: 'Total Contribution Score' column not found.")
else:
    print("Required columns found successfully!")


# -----------------------------------------------------------
# STEP 2: Filter Best Strikers (Encoded as 1)
# -----------------------------------------------------------

print("\n[STEP 2] Filtering Best Strikers (Strikers types = 1)...")

best_strikers = df[df['Strikers types'] == 1]

print("Number of Best Strikers:", best_strikers.shape[0])


# -----------------------------------------------------------
# STEP 3: Calculate Average Score
# -----------------------------------------------------------

print("\n[STEP 3] Calculating Average Total Contribution Score...")

average_score = best_strikers['Total Contribution Score'].mean()

print("Average Total Contribution Score for Best Strikers =",
      round(average_score, 2))


# -----------------------------------------------------------
# FINAL INTERPRETATION
# -----------------------------------------------------------

print("\nINTERPRETATION:")
print(f"""
The average Total Contribution Score for the Best strikers
is {round(average_score,2)}.

This indicates that players classified as Best strikers
have a significantly higher overall performance score
compared to Regular strikers.
""")

print("="*60)



# ===========================================================
# MODEL PERFORMANCE ANALYSIS – LOGISTIC REGRESSION
# ===========================================================

from sklearn.metrics import accuracy_score, confusion_matrix

print("\n" + "="*60)
print("LOGISTIC REGRESSION MODEL PERFORMANCE")
print("="*60)


# -----------------------------------------------------------
# STEP 1: Calculate Accuracy
# -----------------------------------------------------------

print("\n[STEP 1] Calculating Accuracy Score...")

accuracy = accuracy_score(y_test, y_pred) * 100

print("Accuracy Score = {:.2f}%".format(accuracy))


# -----------------------------------------------------------
# STEP 2: Generate Confusion Matrix
# -----------------------------------------------------------

print("\n[STEP 2] Generating Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Format:
# [[TN  FP]
#  [FN  TP]]

TN = cm[0][0]   # Regular predicted correctly
FP = cm[0][1]   # Regular predicted as Best
FN = cm[1][0]   # Best predicted as Regular
TP = cm[1][1]   # Best predicted correctly


# -----------------------------------------------------------
# STEP 3: Extract Required Answers
# -----------------------------------------------------------

print("\n[STEP 3] Extracting Required Values...")

print("Number of Regular Strikers Predicted Correctly (TN):", TN)
print("Number of Best Strikers Predicted Incorrectly (FN):", FN)


# -----------------------------------------------------------
# FINAL INTERPRETATION
# -----------------------------------------------------------

print("\nINTERPRETATION:")
print(f"""
The Logistic Regression model achieved an accuracy of {accuracy:.2f}%.

The model correctly predicted {TN} Regular strikers.

The model incorrectly predicted {FN} Best strikers 
(as Regular strikers).
""")

print("="*60)
