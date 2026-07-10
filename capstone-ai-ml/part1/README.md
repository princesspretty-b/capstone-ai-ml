**Diabetes Dataset Cleaning and Exploratory Data Analysis:**

**Overview**

This project performs data cleaning, preprocessing, exploratory data analysis (EDA), visualization, and statistical analysis on a diabetes dataset obtained from Kaggle. The original dataset was intentionally modified to introduce realistic data quality issues such as incorrect data types, duplicate rows, repetitive categorical values, and corrupted entries. The objective is to demonstrate data cleaning techniques and prepare the dataset for predictive modeling in later parts of the project.

**Dataset Description**

The dataset contains medical measurements commonly used for diabetes prediction.

| **Column**        | **Description**                                     |
| ----------------- | --------------------------------------------------- |
| Pregnant          | Number of pregnancies                               |
| Glucose           | Plasma glucose concentration                        |
| Diastolic_BP      | Diastolic blood pressure (mm Hg)                    |
| Skin_Fold         | Triceps skin fold thickness (mm)                    |
| Serum_Insulin     | 2-Hour serum insulin (mu U/ml)                      |
| BMI               | Body Mass Index                                     |
| Diabetes_Pedigree | Diabetes pedigree function                          |
| Age               | Age of the patient                                  |
| Class             | Diabetes diagnosis (0 = Non-Diabetic, 1 = Diabetic) |
| Hospital          | Hospital Name                                       |

**Dataset Justification**

This dataset is widely used for binary classification problems involving diabetes prediction. It contains a mixture of physiological measurements and demographic information, making it suitable for demonstrating data cleaning, visualization, feature analysis, and statistical techniques commonly applied during data preprocessing.

To simulate real-world data quality issues, the following modifications were intentionally introduced:

- Incorrect data types
- Random string values inserted into numeric columns
- Duplicate records (20%)
- Repetitive categorical values (Hospital column)
- Missing values

These modifications provide opportunities to demonstrate practical preprocessing techniques.

**Data Cleaning**

The following preprocessing steps were performed as per the requirements.

**Task 1: Load Data into Dataframe**

The modified data is loaded in to a Pandas dataframe. The first 5 rows, datatypes and its shaped are populated

**Task 2: Null Value Analysis**

Null values were calculated for every column using both count and percentage using using df.isnull().sum() and (df.isnull().sum()/df.shape\[0\])\*100.

- Table with null count and null percentage for all columns populated
- Null percentage exceeding 20% were printed or reported. This being called out specifically because columns exceeding the threshold would normally require additional investigation or domain-specific treatment at the later parts
- Numeric columns containing less than 20% missing values were imputed using the **median**.

As per our results, Skin_Fold and Serum_Insulin exceeds threshold with null percentage 30.15 and 49.67 respectively

**Task 3: Duplicate detection and removal**

Duplicate rows were identified using df.duplicated().sum() and removed using df.drop_duplicates(). The percentage of missing values before and after duplicate removal was compared. The comparison showed whether the null percentage:

- Increased
- Reduced
- Remained unchanged

As per our results, after the removal of 154 duplicate rows the null value % of Skin_Fold and Serum_Insulin are reduced by 0.59 and 0.98 respectively

**Task 4: Data type correction**

- Columns Age and Glucose contained incorrect data types are stored as object. These columns were converted back to numeric using pd.to_numeric(errors="coerce")
- The repetitive Hospital column was converted to the **category** datatype to reduce memory usage.
- Memory usage before and after conversion was measured using df.memory_usage(deep=True).sum() showing a reduction in memory consumption.

| **Metric**        | **Memory**    |
| ----------------- | ------------- |
| Before Conversion | 171,776 bytes |
| After Conversion  | 62,656 bytes  |
| Memory Saved      | 109,120 bytes |

- Converting the Hospital column to the category data type significantly reduced memory usage because repeated string values are stored efficiently as integer codes with a shared lookup table. Converting the Age and Glucose columns back to numeric also restored their correct data types, making them suitable for statistical analysis and machine learning. Overall, these changes reduced the dataset's memory footprint by approximately 63.5% while ensuring that each column had an appropriate data type for further analysis.

**Task 5: Descriptive statistics and skewness**

Descriptive statistics were computed using df.describe(). Skewness was calculated for every numeric column.

Skewness measures the asymmetry of a distribution.

**Positive skew (right-skewed)**: Most observations are small, with a few unusually large values creating a long right tail. The mean is greater than the median.

**Negative skew (left-skewed):** Most observations are large, with a few unusually small values creating a long left tail. The mean is less than the median.

**Consequence for Missing Value Imputation:** The column with the highest absolute skewness was **Serum_Insulin.** As **Serum_Insulin** is strongly positively skewed, the **median** was selected to impute missing values instead of the mean. The median is less affected by extreme high values and therefore provides a more representative measure of central tendency for this distribution. Using the mean would likely overestimate typical serum insulin values due to the influence of outliers.

**Task 6: Outlier detection with IQR**

Outliers were identified for the following variables using the Interquartile Range (IQR) method.

- Glucose
- BMI

For each variable Q1 , Q3 , IQR , Lower bound , Upper bound were calculated. Rows outside these limits were counted but **not removed**. There was around **8** outliers in **BMI**

The detected outliers may represent genuine medical observations rather than data entry errors.

For example:

- Very high glucose levels can occur in diabetic patients.
- High BMI values may represent obese individuals.

Therefore, the outliers were **retained** during Part 1.

In Part 2, they will be evaluated further. If they negatively affect predictive model performance, they may be capped instead of removed.

**Task 7: Visualizations**

- **line plot :** The line plot shows glucose values across the dataset ordered by row index. It illustrates fluctuations in glucose measurements throughout the dataset.
- **bar chart:** The bar chart compares the average glucose level between diabetic and non-diabetic patients. A higher average glucose level in the diabetic group is expected.
- **histogram** : The histogram shows the distribution of **Serum Insulin** levels across the dataset.

- The distribution is **positively (right) skewed**, with most observations concentrated at lower insulin values and a long tail extending toward higher values.
- A small number of individuals have **very high serum insulin levels**, creating the long right tail and indicating the presence of potential outliers.
- The distribution is **not symmetric** and does not follow a normal distribution.
- Because of this positive skew, the **mean is greater than the median** and is influenced by the extreme high values.

**Implications for Missing Value Imputation**

Since the Serum Insulin distribution is highly right-skewed, using the **mean** to impute missing values could produce values that are higher than what is typical for most patients. Therefore, the **median** is a more appropriate choice for imputing missing values because it is less affected by extreme values and better represents the center of a skewed distribution.

- **scatter plot:** The scatter plot illustrates the relationship between **BMI** and **Glucose** levels, with points colored by **Diabetes Status**.

- The plot shows a **weak to moderate positive relationship** between BMI and Glucose. As BMI increases, glucose levels tend to increase slightly, although the relationship is not very strong.
- The data points are fairly scattered, indicating that **BMI alone is not a strong predictor of glucose levels**. Other factors likely contribute to glucose variation.
- Individuals in the **Diabetic** group are generally concentrated at **higher glucose levels** than those in the **Non-Diabetic** group.
- There is considerable overlap in BMI values between the two groups, suggesting that while BMI is associated with diabetes risk, it does not by itself clearly distinguish diabetic from non-diabetic individuals.

- **box plot** : The box plot compares the distribution of **Glucose** levels between **Diabetic** and **Non-Diabetic** individuals.

- The **median glucose level** is noticeably higher for the **Diabetic** group (around **140 mg/dL**) than for the **Non-Diabetic** group (around **108 mg/dL**). This indicates that diabetic patients generally have higher blood glucose levels.
- The **Diabetic** group has a **wider interquartile range (IQR)**, indicating greater variability in glucose levels.
- The **Non-Diabetic** group has a **narrower spread**, meaning glucose values are more concentrated around the median.
- Several **high-value outliers** are visible in the **Non-Diabetic** group, along with one unusually low outlier. These represent observations that fall outside the typical range according to the IQR rule.
- The **Diabetic** group shows few or no extreme outliers, although the overall distribution covers a wider range of glucose values.

**Task 8: Correlation heat map**

The correlation matrix was computed using df.corr() and visualized with a heat map. The pair of variables with the highest absolute correlation was:

- **Skin_Fold and BMI**
- **Correlation coefficient:** **0.648**

This indicates a **moderately strong positive correlation**, meaning individuals with a higher **Body Mass Index (BMI)** generally tend to have greater **Skin Fold Thickness**.

However, **correlation does not imply causation**. Although BMI and Skin Fold Thickness are related, one variable does not necessarily cause the other.

A plausible alternative explanation is that **both variables measure aspects of body fat or adiposity**:

- **BMI** estimates overall body mass relative to height.
- **Skin Fold Thickness** measures subcutaneous body fat.

Therefore, both variables may increase together because they reflect the same underlying characteristic-**body fat composition**-rather than because one directly causes changes in the other.

Other factors that could influence both variables include:

- **Age**
- **Diet and nutrition**
- **Physical activity level**
- **Genetics**
- **Overall health and metabolic condition**

**Task 9:**

- 1. **Imputation strategy comparison:**

The two most skewed numeric columns were identified using the absolute skewness values.

For each column, both the **mean** and **median** were computed before imputation.

Since both columns were skewed, the **median** was chosen for imputing missing values because:

- **Positive skew:** The mean is pulled upward by extreme high values, while the median better represents the typical observation.
- **Negative skew:** The mean is pulled downward by extreme low values, making the median a more robust measure of central tendency.

After applying fillna() with the median, isnull().sum() confirmed that no missing values remained.

- 2. **Spearman rank correlation:**

The Pearson and Spearman correlation matrices were computed and compared. The three variable pairs with the largest absolute differences between the two correlation measures were:

| **Variable Pair**        | **Pearson** | **Spearman** | **Absolute Difference** |
| ------------------------ | ----------- | ------------ | ----------------------- |
| BMI - Age                | 0.022       | 0.117        | 0.095                   |
| Serum_Insulin - Age      | 0.098       | 0.182        | 0.085                   |
| Pregnant - Serum_Insulin | 0.025       | 0.096        | 0.071                   |

**1\. BMI and Age**

- **Pearson:** 0.022
- **Spearman:** 0.117

Since **|Spearman| > |Pearson|**, the relationship appears to be **monotonic but slightly non-linear**. As age increases, BMI tends to increase slightly in rank order, but the relationship is not proportional enough for Pearson correlation to capture strongly.

**2\. Serum Insulin and Age**

- **Pearson:** 0.098
- **Spearman:** 0.182

Again, **|Spearman| > |Pearson|**, suggesting a **monotonic but non-linear relationship**. Older individuals tend to have somewhat higher serum insulin rankings, although the increase is not linear.

**3\. Pregnant and Serum Insulin**

- **Pearson:** 0.025
- **Spearman:** 0.096

Since **|Spearman| > |Pearson|**, this pair also exhibits a **weak monotonic relationship**. Individuals with a higher number of pregnancies tend to have slightly higher serum insulin rankings, but the relationship is weak and non-linear.

- 3. **Grouped aggregation:**

The dataset was grouped by **Diabetes_Status**, and summary statistics (mean, standard deviation, and count) were calculated for the **Glucose** column.

| **Diabetes Status** | **Mean** | **Standard Deviation** | **Count** |
| ------------------- | -------- | ---------------------- | --------- |
| Diabetic            | 141.87   | 29.47                  | 268       |
| Non-Diabetic        | 110.67   | 24.53                  | 500       |

**(a) Highest Mean and Highest Standard Deviation**

- **Group with the highest mean:** **Diabetic** (Mean = **141.87**)
- **Group with the highest standard deviation:** **Diabetic** (Standard Deviation = **29.47**)

This indicates that diabetic patients have a higher average glucose level than non-diabetic patients and also exhibit greater variability in glucose measurements.

**(b) Is the High Within-Group Standard Deviation a Concern?**

Yes. The diabetic group has a relatively high within-group standard deviation (**29.47**), indicating that glucose levels vary considerably among diabetic individuals. This means that the **Diabetes_Status** feature alone is **not sufficient** to reliably predict an individual's glucose level. Other variables, such as **BMI, Age, Serum_Insulin, and Diabetes_Pedigree**, are also likely to influence glucose levels and should be considered in a predictive model.

**(c) Ratio of the Highest Group Mean to the Lowest Group Mean**

The ratio of the highest group mean to the lowest group mean is:

This means that the average glucose level for the diabetic group is approximately **28% higher** than that of the non-diabetic group.

A ratio of **1.28** indicates a meaningful difference between the two groups, suggesting that **Diabetes_Status** carries useful predictive information. Although the ratio is not extremely large, it shows clear separation between the groups, making this categorical feature valuable when used alongside other clinical variables in predictive modeling.

**Task 10: Save the clean dataset**

The project generates the following outputs:

- diabetes_dataset.csv (corrupted dataset)
- cleaned_data.csv (cleaned dataset)
- Line plot
- Bar chart
- Histogram
- Scatter plot
- Box plot
- Correlation heat map

The cleaned dataset (cleaned_data.csv) will be used as the input for Parts 2 and 3 of the projects.
