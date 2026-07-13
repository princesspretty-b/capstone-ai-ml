**Advanced Modeling - Ensembles, Tuning, and Full ML Pipeline**

**Data Preparation**

**Dataset Loading**

The cleaned dataset generated in **Part 1** was loaded using df = pd.read_csv("cleaned_data.csv"). This dataset contains all preprocessing completed in Part 1, including duplicate removal, missing-value handling, data type correction, and categorical feature creation. And after the preprocessing from part2, the following datasets were available for subsequent modelling tasks.

**Regression**

- X_train_reg_scaled
- X_test_reg_scaled
- y_train_reg
- y_test_reg

**Classification**

- X_train_clf_scaled
- X_test_clf_scaled
- y_train_clf
- y_test_clf

The corresponding unscaled feature matrices (X_train_reg, X_test_reg, X_train_clf, and X_test_clf) were also retained for use in later tasks involving pipelines and hyperparameter tuning. These datasets formed the basis for all predictive models in Part 3.

**Task 1: Decision Tree baseline**

A baseline **Decision Tree Classifier** was trained using the default hyperparameters (max_depth=None). This allows the tree to grow until all leaves are pure or until no further splits are possible.

**Model Performance**

| **Metric**        | **Value**          |
| ----------------- | ------------------ |
| Training Accuracy | **1.000 (100.0%)** |
| Test Accuracy     | **0.695 (69.5%)**  |

**Overfitting Analysis**

The model achieved **100% training accuracy** but only **69.5% test accuracy**, indicating a substantial gap between training and testing performance. This is a clear sign of **overfitting**.

The unrestricted decision tree has effectively memorized the training dataset, including noise and dataset-specific patterns, rather than learning relationships that generalize well to unseen data. As a result, although the model performs perfectly on the training data, its performance drops considerably when evaluated on the test set.

**Why Decision Trees Are High-Variance Models**

Decision Trees are considered **high-variance models** because they construct the tree using a **greedy algorithm**. At each node, the algorithm selects the split that provides the greatest immediate improvement according to an impurity measure (such as Gini impurity or entropy). Once a split has been made, the algorithm **does not revisit or modify earlier decisions**, even if subsequent splits suggest that a different earlier choice might have produced a better overall tree. Because of this greedy construction process, even small changes in the training dataset can produce a very different tree structure. An unconstrained tree is therefore highly sensitive to the training data and can easily overfit by capturing random noise rather than meaningful patterns.

**Inference**

The baseline Decision Tree demonstrates strong evidence of overfitting. Although it achieves perfect performance on the training data, the significant reduction in test accuracy shows that it does not generalize well to new data. In the following task, a **controlled Decision Tree** with constraints on tree depth and minimum samples per split will be trained to reduce overfitting and improve generalization.

**Task 2: Controlled Decision Tree**

A second **Decision Tree Classifier** was trained using the following hyperparameters:

- **max_depth = 5**
- **min_samples_split = 20**

These constraints were introduced to reduce model complexity and improve generalization by preventing the tree from growing excessively deep.

**Model Performance**

| **Metric**        | **Value**         |
| ----------------- | ----------------- |
| Training Accuracy | **0.805 (80.5%)** |
| Test Accuracy     | **0.714 (71.4%)** |

**Role of max_depth**

The max_depth parameter limits the maximum depth of the decision tree. Restricting the tree to **5 levels** prevents it from becoming overly complex and memorizing the training data. While this may slightly reduce training accuracy, it generally improves the model's ability to generalize to unseen data by reducing variance.

**Role of min_samples_split**

The min_samples_split parameter specifies the minimum number of samples required to split an internal node. Setting this value to **20** prevents the model from creating splits based on very small subsets of data, which are often caused by random noise. This results in a simpler and more robust decision tree.

**Comparison with the Unconstrained Decision Tree**

| **Model**                | **Training Accuracy** | **Test Accuracy** | **Train-Test Gap** |
| ------------------------ | --------------------- | ----------------- | ------------------ |
| Default Decision Tree    | **1.000**             | **0.695**         | **0.305**          |
| Controlled Decision Tree | **0.805**             | **0.714**         | **0.091**          |

The unconstrained Decision Tree achieved **100% training accuracy** but only **69.5% test accuracy**, indicating severe overfitting. The model memorized the training data rather than learning patterns that generalize well.

After limiting the tree depth and requiring at least 20 samples before splitting a node, the training accuracy decreased to **80.5%**, while the test accuracy increased slightly to **71.4%**. More importantly, the train-test accuracy gap decreased from **30.5%** to **9.1%**, demonstrating a substantial reduction in overfitting.

**Inference**

The controlled Decision Tree provides a better balance between bias and variance than the unconstrained model. Although it no longer perfectly fits the training data, it performs better on unseen test data and exhibits a much smaller gap between training and testing accuracy. This indicates that the added constraints successfully reduced overfitting and improved the model's generalization performance, making the controlled tree a more reliable model for prediction.

**Task 3: Gini vs Entropy comparison**

Two Decision Tree classifiers were trained using the same tree depth (max_depth = 5) but different splitting criteria to compare their performance.

**Model Performance**

| **Criterion** | **Test Accuracy** |
| ------------- | ----------------- |
| **Gini**      | **0.701 (70.1%)** |
| **Entropy**   | **0.695 (69.5%)** |

The **Gini-based Decision Tree** achieved a slightly higher test accuracy (**70.1%**) than the **Entropy-based Decision Tree** (**69.5%**). The performance difference is small (approximately **0.6 percentage points**), indicating that both impurity measures produce similar predictive performance on this dataset. Based on test accuracy, the Gini criterion performed marginally better.

**Gini Impurity Formula**

The Gini impurity measures the likelihood of incorrectly classifying a randomly selected sample if it were assigned a label according to the class distribution in a node.
            $\text{Gini} = 1 - \sum_{i=1}^{n} p_{i}^2$.

where:

- is the proportion of samples belonging to class .

A lower Gini value indicates a purer node.

**Entropy Formula**

Entropy measures the amount of uncertainty or disorder within a node.

$$
\text{Entropy} = -\sum_{i=1}^{n} p_i \log_2(p_i)
$$

where:

- is the probability of class .

Lower entropy indicates greater purity, while higher entropy indicates a more mixed class distribution.

**Meaning of Gini = 0**

A node has **Gini impurity = 0** when **all samples in that node belong to the same class**. Such a node is called a **pure node** because there is no uncertainty in classification.

For example:

- 100% Diabetic, 0% Non-Diabetic, or
- 100% Non-Diabetic, 0% Diabetic.

Since every sample belongs to a single class, no further splitting is necessary.

**Inference**

Both Gini and Entropy aim to create the purest possible child nodes, but they use different mathematical measures to evaluate impurity. Gini impurity is computationally simpler because it does not require logarithmic calculations, making it slightly faster to compute. Entropy uses information gain based on information theory and may sometimes produce different split decisions.

For this diabetes dataset, both criteria yielded very similar results, suggesting that the choice of impurity measure has only a minor effect on classification performance. The Gini criterion achieved the highest test accuracy and would therefore be the preferred choice.

**Task 4: Random Forest**

A **Random Forest Classifier** was trained using the following hyperparameters:

- **n_estimators = 100** (100 decision trees)
- **max_depth = 10**
- **random_state = 42**

The purpose of using Random Forest was to improve generalization compared with a single Decision Tree by combining multiple trees and reducing variance.

**Model Performance**

| **Metric**        | **Value**         |
| ----------------- | ----------------- |
| Training Accuracy | **0.976 (97.6%)** |
| Test Accuracy     | **0.714 (71.4%)** |
| ROC-AUC           | **0.767**         |

The Random Forest achieved a high training accuracy of **97.6%** and a lower test accuracy of **71.4%**, indicating some degree of overfitting. However, compared with the unrestricted Decision Tree, the performance is more stable because the ensemble approach reduces the sensitivity of the model to individual training samples.

The ROC-AUC score of **0.767** indicates that the model has a reasonable ability to distinguish between diabetic and non-diabetic patients. An AUC of 0.5 represents random guessing, while an AUC closer to 1.0 represents stronger separation between classes.

**Top Five Feature Importances**

| **Rank** | **Feature**       | **Importance** |
| -------- | ----------------- | -------------- |
| 1        | BMI               | **0.201972**   |
| 2        | Age               | **0.162440**   |
| 3        | Diabetes_Pedigree | **0.151472**   |
| 4        | Serum_Insulin     | **0.146821**   |
| 5        | Skin_Fold         | **0.104489**   |

The Random Forest identified **BMI** as the most influential feature, contributing approximately **20.2%** of the total feature importance. Age, Diabetes Pedigree, Serum Insulin, and Skin Fold were also important predictors for classification.

**Feature Importance Interpretation**

Random Forest calculates feature importance by measuring the **average reduction in Gini impurity** caused by each feature across all splits and across all trees in the forest.

When a feature is used to split a decision node, the reduction in impurity is calculated. A feature that consistently creates purer child nodes across many trees receives a higher importance score. The final feature importance is the normalized average of these impurity reductions.

For this model, **BMI has the highest importance**, meaning that splits based on BMI contributed the greatest reduction in classification uncertainty across the forest.

Random Forest feature importance differs from a **Linear Regression coefficient**. A linear regression coefficient represents the expected change in the continuous target variable for a one-unit increase in a feature while keeping other variables constant. It provides both direction (positive or negative) and magnitude of a linear relationship.

In contrast, Random Forest feature importance only measures how useful a feature is for making predictions. It does not indicate whether the feature increases or decreases the probability of diabetes. A high importance score means the feature contributes strongly to prediction, but it does not describe the direction of its relationship with the target.

**Bagging and Variance Reduction**

Random Forest uses **bagging (Bootstrap Aggregation)** to improve model stability and reduce overfitting.

During training, each decision tree is created using a **bootstrap sample**, where observations are randomly selected from the training data with replacement. This means that each tree receives a slightly different subset of the training examples.

Additionally, at every split in a tree, Random Forest considers only a random subset of features (approximately **√(number of features)**) instead of all available features. This introduces additional diversity because different trees learn different patterns from different feature combinations.

After all trees are trained, their predictions are combined using majority voting for classification. Because individual trees have different errors, averaging their predictions reduces variance and produces a more reliable model than a single deep decision tree.

This explains why Random Forest generally achieves better generalization performance than an unrestricted Decision Tree, which can memorize training data and suffer from high variance.

**Task 4a: Gradient Boosting**

A **Gradient Boosting Classifier** was trained using the following hyperparameters:

- **n_estimators = 100**
- **learning_rate = 0.1**
- **max_depth = 3**
- **random_state = 42**

Gradient Boosting is an ensemble learning technique that builds decision trees sequentially. Each new tree attempts to correct the errors made by previous trees by focusing more on difficult-to-classify samples. This iterative learning process allows the model to improve prediction performance while controlling model complexity.

**Model Performance**

| **Metric**        | **Value**         |
| ----------------- | ----------------- |
| Training Accuracy | **0.902 (90.2%)** |
| Test Accuracy     | **0.727 (72.7%)** |
| ROC-AUC           | **0.781**         |

The Gradient Boosting model achieved a training accuracy of **90.2%** and a test accuracy of **72.7%**, resulting in a relatively small train-test gap of approximately **17.5%**. Compared with the unrestricted Decision Tree, which achieved perfect training accuracy but lower generalization performance, Gradient Boosting demonstrates better control over overfitting.

The ROC-AUC score of **0.781** indicates that the model has good ability to distinguish between diabetic and non-diabetic patients. The model can correctly rank a randomly selected positive case higher than a randomly selected negative case approximately **78.1% of the time**.

**Task 4b: Feature ablation study**

A feature ablation study was performed using the feature importance scores obtained from the Random Forest model. The objective was to determine whether the least important features contributed meaningful predictive information or whether they added unnecessary complexity to the model.

The original Random Forest model was trained using:

- n_estimators = 100
- max_depth = 10
- random_state = 42

The five features with the lowest feature importance scores were removed, and a second Random Forest model was trained using the same hyperparameters. Both models were evaluated using ROC-AUC on the same test dataset.

**Five Least Important Features Removed**

| **Feature**         | **Importance** |
| ------------------- | -------------- |
| Hospital_Gleneagals | 0.011879       |
| Hospital_Kavery     | 0.012219       |
| Hospital_Fortis     | 0.014747       |
| Hospital_MMM        | 0.015066       |
| Pregnant            | 0.087224       |

The hospital-related categorical features had the lowest importance scores, indicating that the hospital identifier provided very little predictive information. The Pregnant feature also had relatively low contribution compared with the other clinical measurements.

**ROC-AUC Comparison**

| **Model**                                  | **ROC-AUC** |
| ------------------------------------------ | ----------- |
| Full Random Forest (All Features)          | 0.767407    |
| Reduced Random Forest (5 Features Removed) | 0.771759    |

**Interpretation of Feature Ablation Results**

The reduced Random Forest achieved a ROC-AUC of **0.771759**, which is slightly higher than the full model ROC-AUC of **0.767407**.

Since removing the five lowest-importance features did not decrease model performance and actually produced a small improvement, these features appear to be **largely uninformative or slightly noisy** for this prediction task. The removed features were not providing additional predictive value and may have introduced unnecessary variation into the model.

The improvement suggests that the model was able to generalize slightly better after removing less useful inputs.

**Production Deployment Consideration**

A reduced-feature model can provide several advantages in a production environment:

- Lower data collection requirements
- Reduced preprocessing complexity
- Faster prediction time
- Easier model monitoring and maintenance
- Lower risk of missing values from unnecessary features

However, feature reduction should only be performed when predictive performance remains within an acceptable range. In this case, removing five features improved ROC-AUC from **0.767407 to 0.771759**, meaning the simpler model provides equal or better predictive performance while requiring fewer inputs.

Therefore, the reduced Random Forest would be preferable for deployment because it reduces operational complexity without sacrificing model quality.

**Task 5: Cross-validated comparison**

The ROC-AUC metric was used because it evaluates the ability of each model to distinguish between diabetic and non-diabetic classes across different classification thresholds.

| **Model**                | **Mean 5-Fold AUC** | **Std AUC** |
| ------------------------ | ------------------- | ----------- |
| Logistic Regression      | 0.756020            | 0.020266    |
| Controlled Decision Tree | 0.716705            | 0.059564    |
| Random Forest            | 0.759456            | 0.023668    |
| Gradient Boosting        | **0.770217**        | 0.028677    |

**Interpretation of Cross-Validation Results**

The cross-validation results show that **Gradient Boosting achieved the highest average ROC-AUC score (0.770217)** among all evaluated models. This indicates that Gradient Boosting provided the strongest overall ability to distinguish between diabetic and non-diabetic cases.

Random Forest achieved the second-highest mean AUC (**0.759456**), followed closely by Logistic Regression (**0.756020**). The Controlled Decision Tree produced the lowest mean AUC (**0.716705**) and also had the highest variation between folds (**standard deviation = 0.059564**), suggesting that its performance was less stable across different subsets of the dataset.

The standard deviation values indicate how much the model performance changes depending on the training and validation split. Lower standard deviation generally indicates more consistent model behavior. Logistic Regression showed the most stable performance with a standard deviation of **0.020266**, while the Controlled Decision Tree showed greater sensitivity to the data split.

**Why Cross-Validation Provides a More Reliable Estimate**

A single train-test split evaluates a model using only one specific division of the dataset. The measured performance can be affected by random variation in which samples are placed into the training and testing groups. This can lead to an overly optimistic or pessimistic estimate of model performance.

Five-fold stratified cross-validation reduces this limitation by dividing the dataset into five different training-validation combinations while maintaining the same class distribution in each fold. Each model is trained five times and evaluated on different unseen subsets of data.

The final AUC score is calculated as the average across all five folds, providing a more reliable estimate of expected real-world performance. The standard deviation also shows the consistency of the model across different samples.

Based on the cross-validation results, **Gradient Boosting is the preferred model** because it achieved the highest mean ROC-AUC (**0.770217**) while maintaining reasonable stability across folds.

Although Random Forest performed competitively, Gradient Boosting provided better average discrimination ability. Logistic Regression showed stable performance but had a slightly lower AUC, while the Controlled Decision Tree had weaker predictive performance and higher variability.

For deployment, Gradient Boosting would be recommended because it provides the best balance between predictive performance and generalization ability on this diabetes classification problem.

**Task 6: Hyperparameter tuning with GridSearchCV**

**Random Forest Hyperparameter Optimization**

A Random Forest classifier was optimized using GridSearchCV with a preprocessing pipeline consisting of:

- **SimpleImputer(strategy='median')**
  - Missing numerical values are replaced using the median value calculated from the training data.
- **StandardScaler()**
  - Features are standardized before model training.
- **RandomForestClassifier(random_state=42)**
  - Used as the classification algorithm.

The pipeline was trained using the original unscaled X_train_clf and y_train_clf from Part 2. The preprocessing steps were performed inside the pipeline to ensure that information from the test data was not used during training.

**Hyperparameter Search Space**

The following Random Forest parameters were evaluated:

| **Parameter**    | **Values**   |
| ---------------- | ------------ |
| n_estimators     | 50, 100, 200 |
| max_depth        | 5, 10, None  |
| min_samples_leaf | 1, 5         |

The total number of model configurations evaluated was 3 \* 3 \* 2 = 18

Using 5-fold Stratified Cross Validation : 18 \* 5 = 90

Therefore, **90 Random Forest models were trained and evaluated** during GridSearchCV.

**Best Grid Search Model**

| **Parameter**                               | **Best Value** |
| ------------------------------------------- | -------------- |
| Number of Trees (n_estimators)              | 200            |
| Maximum Depth (max_depth)                   | 5              |
| Minimum Samples per Leaf (min_samples_leaf) | 1              |
| Best Cross-Validation ROC-AUC               | 0.7796         |

The best-performing Random Forest configuration used **200 decision trees with a maximum depth of 5**. Limiting the depth helps control model complexity and reduces overfitting while increasing the number of trees improves stability by averaging predictions from multiple learners.

**Grid Search vs Randomized Search**

Grid Search performs an exhaustive evaluation of all parameter combinations provided in the search space. This guarantees that the best combination within the defined range is found, but computational cost increases as more parameters and values are added.

Randomized Search evaluates only a selected number of randomly sampled parameter combinations. It can explore a larger parameter space with fewer model evaluations and is usually more efficient for complex machine learning problems.

For this project, GridSearchCV was appropriate because the parameter space was relatively small, requiring only 18 configurations. For larger datasets or models with many hyperparameters, RandomizedSearchCV would provide faster optimization while still finding competitive parameter settings.

**Task 7: Manual learning curve**

The best Random Forest pipeline obtained from GridSearchCV was retrained using progressively larger subsets of the training dataset (20%, 40%, 60%, 80%, and 100%). For each training size, the model was evaluated using ROC-AUC on both the subset used for training and the fixed test dataset.

**Learning Curve Results**

| **Training Fraction** | **Training AUC** | **Test AUC** |
| --------------------- | ---------------- | ------------ |
| 20%                   | 0.9985           | 0.7539       |
| 40%                   | 0.9731           | 0.7798       |
| 60%                   | 0.9489           | 0.7735       |
| 80%                   | 0.9271           | 0.7867       |
| 100%                  | 0.9161           | 0.7915       |

**Training AUC Interpretation**

The training AUC decreased as the training dataset size increased:

- 20% training data: **0.9985**
- 100% training data: **0.9161**

This decrease is expected for a high-variance model such as Random Forest. When trained on a small subset of data, the model can memorize specific patterns and achieve extremely high training performance. As more samples are introduced, the model encounters more variation and becomes less likely to memorize individual examples, resulting in a lower but more realistic training AUC.

**Test AUC Interpretation**

The test AUC generally increased as more training data was provided:

- 20% training data: **0.7539**
- 100% training data: **0.7915**

Although there were small fluctuations between training sizes (for example, the 60% training subset produced slightly lower AUC than the 40% subset), the overall trend shows improvement as additional training data was added.

This suggests that increasing the amount of available training data helps the model learn more generalizable patterns and improves performance on unseen data.

**Conclusion**

The learning curve indicates that the model is **currently data-limited rather than capacity-limited**.

The test AUC continued increasing when moving from 80% training data (**0.7867**) to the full training dataset (**0.7915**). Since performance has not completely plateaued, collecting additional labeled diabetes data may provide further improvements.

The model does not appear to be limited primarily by model complexity because the Random Forest has already achieved strong generalization performance. Future improvements are more likely to come from:

- collecting more patient records,
- improving feature quality,
- adding clinically relevant variables,
- or performing additional feature engineering.

A larger dataset may also help reduce the remaining gap between training and test AUC and improve model robustness in production.

**Task 8: Serialize the best model**

- Saving the best pipeline as best_model.pkl using the best_pipeline object created from GridSearchCV.in the repository
- Reloading the saved best_model.pkl and a prediction was made on 2 sample data, the output will be a Diabetes predicted class and diabetes probability.

**Task 9: Summary comparison table**

The following table compares all classification models from Parts 2 and 3 using:

- **5-fold Cross-Validation Mean ROC-AUC**
- **5-fold Cross-Validation Standard Deviation**
- **Test-set ROC-AUC**

| **Model**                                      | **5-Fold CV Mean AUC** | **5-Fold CV Std AUC** | **Test-set AUC** |
| ---------------------------------------------- | ---------------------- | --------------------- | ---------------- |
| Logistic Regression (C=1.0)                    | 0.7560                 | 0.0203                | 0.7822           |
| Controlled Decision Tree (max_depth=5)         | 0.7167                 | 0.0596                | 0.7013           |
| Random Forest (n_estimators=100, max_depth=10) | 0.7595                 | 0.0237                | 0.7674           |
| Gradient Boosting                              | 0.7702                 | 0.0287                | 0.7810           |
| Tuned Random Forest (GridSearchCV)             | 0.7796                 | 0.0257                | 0.7914           |

**Final Model Recommendation**

The recommended model for the client is the **tuned Random Forest model optimized using GridSearchCV**. It achieved the highest cross-validation ROC-AUC score (**0.7796**), indicating strong and consistent ability to distinguish between diabetic and non-diabetic patients across multiple validation splits. The model also provides feature importance scores, which improve interpretability by showing which patient characteristics contribute most to predictions. Compared with a single decision tree, the Random Forest approach reduces variance through ensemble averaging and provides more reliable generalization performance. The final model provides a good balance between predictive performance, robustness, and practical deployment considerations.
