# Import Libraries and get sample diabetes dataset from Kaggle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

plt.style.use("seaborn-v0_8")
sns.set_theme(style="whitegrid")

path = kagglehub.dataset_download("bhavyamotiyani/diabetes-dataset-with-missing-value-for-practice")
file_path = os.path.join(path, "Diabetes Missing Data.csv")

print("Path to dataset files:", file_path)

# Read the ideal data from Kaggle for Diabetes and intentionally corrupt the data to support 
# the following, so the data resemblems the real world data issues:
# a)Data Type change b)Inclusion of categorical column, c) Duplicate 20% od data d)Shuffle all rows

df_d = pd.read_csv(file_path)

# Duplicate 20% of the dataset
np.random.seed(42)
# Change Age column from int to object
df_d["Age"] = df_d["Age"].astype(object)

# Randomly replace 15 values with strings
age_idx = np.random.choice(df_d.index, 15, replace=False)
df_d.loc[age_idx[:5], "Age"] = "Unknown"
df_d.loc[age_idx[5:10], "Age"] = "NA"
df_d.loc[age_idx[10:], "Age"] = "Thirty"

# Change Glucose column from float to object
df_d["Glucose"] = df_d["Glucose"].astype(object)

glucose_idx = np.random.choice(df_d.index, 10, replace=False)
df_d.loc[glucose_idx[:5], "Glucose"] = "Missing"
df_d.loc[glucose_idx[5:], "Glucose"] = "?"

# Add a repetitive string column
df_d["Hospital"] = np.random.choice(
    ["Apollo", "Fortis", "Kavery", "MMM", "Gleneagals"],
    size=len(df_d)
)
print(f"\nData Types\n-----------\n{df_d.dtypes}")
# Duplicate 20% of the dataset
duplicate_fraction = 0.2
duplicate_rows = df_d.sample(frac=duplicate_fraction, random_state=42)
df = pd.concat([df_d, duplicate_rows], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("diabetes_dataset.csv", index=False)
new_file_path = "diabetes_dataset.csv"
print(f"\nShape : {df.shape}")

# Task 1: Load the dataset into a pandas DataFrame using pd.read_csv() (or equivalent). 
# Print the first five rows, the column data types (.dtypes), and the DataFrame shape.

df = pd.read_csv(new_file_path)

# Display the first 5 rows
print(df.head())
# Column data types
print(f"\nData Types\n-----------\n{df.dtypes}")
# Dataframe Shape
print(f"\nShape : {df.shape}")

'''Task 2: Null value analysis
a) Compute the count and percentage of missing values in every column using df.isnull().sum() 
and (df.isnull().sum() / df.shape[0]) * 100.'''

null_count = df.isnull().sum()
null_percent = (df.isnull().sum()/df.shape[0])*100

null_table = pd.DataFrame({
    "Missing Count":null_count,
    "Missing %":null_percent
})
print("Missing value Table\n--------------------\n", null_table)

'''
b) Report which columns exceed a 20% null rate. For columns below 20% nulls, fill numeric columns with the column median 
using fillna(df[col].median()).'''

print("\ncolumns exceed a 20% null rate\n-------------------------------")
print(null_table[null_table["Missing %"]>20])

# For columns below 20% nulls, fill numeric columns with the column median using fillna(df[col].median()). 
for col in df.columns:
    if df[col].isnull().mean()*100 <20:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())

print("\nAfter filling median\n------------------------")
print(df.isnull().sum())

'''Task 3: Duplicate detection and removal

a) Use df.duplicated().sum() to count duplicates. Remove them with df.drop_duplicates().
b) Report how many rows were removed and whether the removal changes any column's null percentage.'''

duplicates = df.duplicated().sum()
print("Duplicate Rows Present =",duplicates)
before = df.shape[0]
null_count_before = df.isnull().sum()
null_percent_before = (df.isnull().sum()/df.shape[0])*100
df = df.drop_duplicates()
after = df.shape[0]
null_count_after = df.isnull().sum()
null_percent_after = (df.isnull().sum()/df.shape[0])*100
print("Duplicate Rows Removed =",before-after)
comparision_table = pd.DataFrame({
    "Null Count Before":null_count_before,
    "Null % Before":null_percent_before,
    "Null Count After":null_count_after,
    "Null % After":null_percent_after})

# Difference in null %
comparision_table["Change (%)"] = (
    comparision_table["Null % After"] - comparision_table["Null % Before"]
).round(2)

# Status
comparision_table["Status"] = np.where(
    comparision_table["Change (%)"] > 0, "Increased",
    np.where(comparision_table["Change (%)"] < 0, "Reduced", "No Change")
)
print("\nNull % Before vs After\n-----------------------\n",comparision_table)

'''Task 4: Data type correction

a) Identify at least one column whose inferred dtype is incorrect (e.g., a numeric column stored as object). Convert it using astype() or pd.to_numeric() with errors='coerce'.
b) Convert at least one repetitive string column to category dtype.
c) Report memory usage before and after the conversion using df.memory_usage(deep=True).sum().'''

# Memory usage before conversion
memory_before = df.memory_usage(deep=True).sum()
print(f"Memory Usage Before: {memory_before:,} bytes")

# Check current data types
print("\nData Types Before:")
print(df.dtypes)

# Correct incorrect dtype (numeric stored as object)
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Glucose"] = pd.to_numeric(df["Glucose"], errors="coerce")

# Convert repetitive string column to category
df["Hospital"] = df["Hospital"].astype("category")

# Memory usage after conversion
memory_after = df.memory_usage(deep=True).sum()
print(f"\nMemory Usage After: {memory_after:,} bytes")
print(f"Memory Saved: {memory_before - memory_after:,} bytes")

# Check updated data types
print("\nData Types After:")
print(df.dtypes)
# Correcting the corrupted rows as this wold have been take care by the code if task 4 is done before task 2
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Glucose"] = df["Glucose"].fillna(df["Glucose"].median())

'''Task 5: Descriptive statistics and skewness

a) Call df.describe() on all numeric columns.
b) For each numeric column, compute df[col].skew().
c) Identify and name the column with the highest absolute skewness.
d) Explain in the README what positive vs negative skew means for that 
column's distribution and what consequence it has for imputing missing values with the mean.'''

# Descriptive Statistics
print("Descriptive Statistics")
print("-" * 60)
print(df.describe())

# Skewness of Numeric Columns
skewness = df.select_dtypes(include="number").skew()
print("\nSkewness of Numeric Columns")
print("-" * 60)
print(skewness)

# Column with Highest Absolute Skewness
highest_skew_col = skewness.abs().idxmax()
highest_skew_value = skewness[highest_skew_col]

print("\nColumn with Highest Absolute Skewness")
print("-" * 60)
print(f"Column : {highest_skew_col}")
print(f"Skewness : {highest_skew_value:.3f}")

'''Task 6: Outlier detection with IQR

a) For at least two numeric columns, compute Q1 (df[col].quantile(0.25)), Q3 (df[col].quantile(0.75)), 
IQR = Q3 − Q1, lower bound = Q1 − 1.5 × IQR, and upper bound = Q3 + 1.5 × IQR.
b) Count the number of rows that fall outside these bounds. Do not drop the outliers — instead, document them and state in the 
README whether you will cap them, retain them, or handle them differently in Part 2, and why.'''

# Select columns for outlier detection
columns = ["Glucose", "BMI"]   # Change these to any numeric columns you want

print("Outlier Detection using IQR Method")
print("-" * 70)

outlier_summary = []

for col in columns:
    # Calculate quartiles and IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate lower and upper bounds
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    # Identify outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    outlier_summary.append({
        "Column": col,
        "Q1": round(Q1, 2),
        "Q3": round(Q3, 2),
        "IQR": round(IQR, 2),
        "Lower Bound": round(lower_bound, 2),
        "Upper Bound": round(upper_bound, 2),
        "Outlier Count": len(outliers)
    })

# Display summary table
outlier_df = pd.DataFrame(outlier_summary)

print("\nOutlier Summary")
print(outlier_df)

print("\nOutlier Rows:")
print(outliers[[col]])

'''Task 7 : Visualizations (all five types required)

a) A line plot of one numeric variable sorted by row index or time column (use plt.plot()). Add a title, x-label, and y-label.'''

# Create a categorical column from Class
df["Diabetes_Status"] = df["Class"].map({0: "Non-Diabetic", 1: "Diabetic"})

plt.figure(figsize=(10,5))
plt.plot(df.index, df["Glucose"], color="blue")

plt.title("Glucose Levels by Row Index")
plt.xlabel("Row Index")
plt.ylabel("Glucose")
plt.savefig("part1/plots/glucose_plot.png", dpi=300, bbox_inches="tight")
plt.show()

'''b) A bar chart comparing the mean of one numeric column across categories of one 
categorical column (use plt.bar() or df.groupby().mean().plot.bar()). Add a title and axis labels.'''

mean_glucose = df.groupby("Diabetes_Status")["Glucose"].mean()

plt.figure(figsize=(6,5))
plt.bar(mean_glucose.index, mean_glucose.values,
        color=["skyblue","orange"])

plt.title("Average Glucose by Diabetes Status")
plt.xlabel("Diabetes Status")
plt.ylabel("Mean Glucose")
plt.savefig("part1/plots/Avg_glucose.png", dpi=300, bbox_inches="tight")
plt.show()

'''c) A histogram of the most skewed numeric column identified in Task 5 (use sns.histplot() with bins=20). 
Add a title and describe the shape of the distribution in the README.'''

plt.figure(figsize=(8,5))

sns.histplot(df["Serum_Insulin"], bins=20, kde=True)

plt.title("Distribution of Serum Insulin")
plt.xlabel("Serum Insulin")
plt.ylabel("Frequency")
plt.savefig("part1/plots/Serum_Insulin.png", dpi=300, bbox_inches="tight")
plt.show()

'''d) A scatter plot between two numeric columns that you expect to be correlated (use sns.scatterplot()). 
Add a title and interpret the direction and approximate strength of the relationship in the README.'''

plt.figure(figsize=(8,5))

sns.scatterplot(data=df,
                x="BMI",
                y="Glucose",
                hue="Diabetes_Status")

plt.title("BMI vs Glucose")
plt.xlabel("BMI")
plt.ylabel("Glucose")
plt.savefig("part1/plots/BMI_Glucose.png", dpi=300, bbox_inches="tight")
plt.show()

'''e) A box plot of a numeric column split by a categorical column (use sns.boxplot()). 
Add a title and describe any visible differences in median and spread across categories.'''

plt.figure(figsize=(6,5))

sns.boxplot(data=df,
            x="Diabetes_Status",
            y="Glucose")

plt.title("Glucose Distribution by Diabetes Status")
plt.xlabel("Diabetes Status")
plt.ylabel("Glucose")
plt.savefig("part1/plots/Gulcose_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

'''Task 8: Correlation heat map

a) Compute the correlation matrix of all numeric columns using df.corr(). 
Visualize it with a heat map (use sns.heatmap() with annot=True).'''

# Select numeric columns only
numeric_df = df.select_dtypes(include="number")

# Compute correlation matrix
corr_matrix = numeric_df.corr()

print("Correlation Matrix")
print("-" * 50)
print(corr_matrix)

# Plot heat map
plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heat Map of Numeric Variables")
plt.savefig("part1/plots/Correlation_Heat_Map.png", dpi=300, bbox_inches="tight")
plt.show()

'''
b) Identify the pair of variables with the highest absolute correlation.
c) In the README, explain whether this correlation might indicate a causal relationship or whether a third variable 
could explain it, and name at least one plausible alternative explanation.'''

# Remove self-correlations (diagonal values)
corr_pairs = corr_matrix.abs().unstack()

# Remove duplicate pairs
corr_pairs = corr_pairs[corr_pairs < 1]

# Find highest correlation pair
highest_corr_pair = corr_pairs.idxmax()
highest_corr_value = corr_matrix.loc[
    highest_corr_pair[0], highest_corr_pair[1]
]

print("Highest Absolute Correlation Pair:")
print("-" * 36)
print(f"{highest_corr_pair[0]} and {highest_corr_pair[1]}")
print(f"Correlation: {highest_corr_value:.3f}")

'''Task 9: a. Imputation strategy comparison.

i) For the two numeric columns with the highest absolute skewness identified in Task 5, 
compute both the column mean (df[col].mean()) and the column median (df[col].median()) before any imputation is 
applied to those columns. Print both values side by side for each column.

ii) In the README, state which statistic you chose for imputation and justify the choice based on the skewness direction: 
for a positively skewed column the mean is pulled upward by extreme high values, making the median a more 
representative central tendency; for a negatively skewed column the mean is pulled downward by extreme low values, 
again making the median more representative.

iii) Apply your chosen strategy using fillna() to any remaining nulls in those two columns and 
confirm with isnull().sum() that no nulls remain.'''

# Compute skewness for numeric columns
skewness = df.select_dtypes(include='number').skew()

# Two columns with highest absolute skewness
top2_skewed = skewness.abs().sort_values(ascending=False).head(2)

print("Two Most Skewed Columns:")
print(top2_skewed)

#Apply both the column mean before imputation
print("\nMean vs Median Before Imputation")
print("-"*33)

for col in top2_skewed.index:
    print(f"\nColumn: {col}")
    print(f"Mean   : {df[col].mean():.2f}")
    print(f"Median : {df[col].median():.2f}")

# Impute missing values
for col in top2_skewed.index:

    if skewness[col] > 0:
        print(f"{col}: Positively skewed -> Imputing with Median")
        df[col] = df[col].fillna(df[col].median())

    elif skewness[col] < 0:
        print(f"{col}: Negatively skewed -> Imputing with Median")
        df[col] = df[col].fillna(df[col].median())

    else:
        print(f"{col}: Approximately symmetric -> Imputing with Mean")
        df[col] = df[col].fillna(df[col].mean())

print("\nRemaining Null Values")

for col in top2_skewed.index:
    print(col, ":", df[col].isnull().sum())

'''Task 9: b) Spearman rank correlation.

i) Use df.corr(method='spearman') and compare the result to your Pearson matrix from Task 8 — 
a column pair where |Spearman| > |Pearson| indicates a consistent but non-linear relationship between those two variables
ii) Compute the Spearman rank correlation matrix for all numeric columns using df.corr(method='spearman').
iii) Identify the three column pairs where the absolute difference between the Spearman correlation and 
the Pearson correlation (from Task 8) is largest. Print both matrices and a difference table 
showing |Spearman − Pearson| for each pair.
iv) In the README, explain for each of the three identified pairs: 
(a) whether the relationship appears to be monotonic but non-linear (if |Spearman| > |Pearson|, 
indicating the variables move together consistently but not proportionally) or approximately linear 
(if |Pearson| ≥ |Spearman|); and (b) which correlation measure you will rely on 
for feature-selection guidance in Part 2 and why.'''

#Pearson Matrix
pearson_corr = df.select_dtypes(include='number').corr()
print("Pearson Correlation")
print(pearson_corr)

#Spearman Matrix
spearman_corr = df.select_dtypes(include='number').corr(method='spearman')
print("\nSpearman Correlation")
print(spearman_corr)

#Difference
difference = (spearman_corr - pearson_corr).abs()
print("\nAbsolute Difference")
print(difference)

#Top 3 Pairs
pairs = []

cols = difference.columns

for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        pairs.append([
            cols[i],
            cols[j],
            pearson_corr.iloc[i,j],
            spearman_corr.iloc[i,j],
            difference.iloc[i,j]
        ])

difference_table = pd.DataFrame(
    pairs,
    columns=[
        "Variable 1",
        "Variable 2",
        "Pearson",
        "Spearman",
        "|Difference|"
    ]
)

difference_table = difference_table.sort_values(
    by="|Difference|",
    ascending=False
)

print("\nTop 3 Largest Differences")
print(difference_table.head(3))

'''Task 9 : Grouped aggregation.

Choose one categorical column and one numeric column from the cleaned dataset. 
Compute df.groupby(categorical_col)[numeric_col].agg(['mean', 'std', 'count']) and print the result.'''

group_stats = df.groupby("Diabetes_Status")["Glucose"].agg(
    ["mean","std","count"]
)
print(group_stats)
# Highest Mean
highest_mean = group_stats["mean"].idxmax()
print("\nHighest Mean:", highest_mean)

# Highest Standard Deviation
highest_std = group_stats["std"].idxmax()
print("\nHighest Std Dev:", highest_std)

# Mean Ratio
ratio = group_stats["mean"].max() / group_stats["mean"].min()
print(f"\nMean Ratio = {ratio:.2f}")

'''Task 10: Save the clean dataset to a file named cleaned_data.csv using df.to_csv('cleaned_data.csv', index=False).'''

df.to_csv("cleaned_data.csv", index=False)
print("Cleaned dataset saved successfully as 'cleaned_data.csv'")