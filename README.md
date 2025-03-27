# Drug Response Prediction in Rheumatoid Arthritis 

## Project Overview
This project aims to predict patient response to treatment using gene expression data and patient meta data.
I developed a machine learning classification models to identify responders versus non-responders based on: 
1. patient meta data (DAS28, Gender etc,.)
2. gene expression data 


## Repository Structure
drug-response-prediction/
```markdown
drug-response-prediction/
├── data/                  # Raw data files
├── scripts/               # Python scripts for preprocessing and modeling
├── plots/                 # Generated visualizations and results
├── requirements.txt       # Package dependencies
└── README.md              # This file
```

## Key Insights from Exploratory Data Analysis(EDA)

### Meta Data Analysis
- **Balanced dataset**: 24 responders and 22 non-responders, providing a good foundation for unbiased model training, indicating a relatively balanced dataset (plot path: 'plots/1_1_response_distribution.png')

- **Gender distribution**: No significant difference in response rates between genders, gender may not be a strong predictor of treatment outcome (plot path: 'plots/1_2_gender_response.png')

- **DAS28 scores**: Non-responders showed slightly higher median DAS28 scores with more variability, disease severity may influence treatment outcomes (plot path: 'plots/1_3_das28_by_response.png')

### Gene Expression Analysis
- **Diverse expression patterns**: The analyzed genes showed distinct distribution patterns (I randomly sampled 4 genes to examine their expression distribution):
  - 1007_s_at and 121_at: Highly skewed toward lower expression levels
  - 1053_at: More normally distributed with moderate expression levels
  - 117_at: Normally distributed with higher expression levels
(plot path: 'plots/1_4_gene_expressions_distribution.png')

- **Correlation analysis**: 
  - Moderate negative correlation (-0.24) between 'DAS28' and gene '1007_s_at'
  - Strong positive correlation (0.41) between genes '1053_at' and '117_at'
  - Generally weak correlations between DAS28 and most gene markers
(plot path: '1_5_correlation_matrix.png')



### Handeling missing values

| Feature  | Missing Values | Percentage (%) |
|----------|----------------|----------------|
| Response | 40             | 46.51          |
| DAS28    | 6              | 6.98           |

- **Response (target variable)**
I noticed that 46.5% of the target variable (Response) values are missing.
There are several possible approaches to handle this issue:

1.Removing rows with missing values in Response column
2.Imputing the missing values using a model
3.Treating missing values as a separate category

Since I don’t have domain knowledge about why these values are missing, I will first try to remove records with a missing target variable and evaluate the model results.
If the models fail to generalize and achieve sufficient accuracy, I will consider the third approach (treating missing values as a separate category), and only as a last resort, explore imputation using a model.

- **DAS28 (explanatory variable)**
There are 7% of missing values in these variable.
The skewness of DAS28 is -0.25, which indicates that the data is approximately normally distributed and negatively skewed.
Since the absolute value of the skewness is less than 0.5 and the skewness is negative, I decided to impute the missing values using the mean

### Handeling outliers values

** I took a sample of 1000 numeric column just for example of dealing with ouliers** 

| Feature       | lower_threshold | upper_threshold | has_outliers |
|--------------|-----------------|-----------------|--------------|
| DAS28        | 2.672500        | 7.872500        | 0            |
| 1007_s_at    | 2.161545        | 2.736109        | 1            |
| 1053_at      | 4.151957        | 6.923983        | 1            |
| 117_at       | 6.936775        | 10.214560       | 1            |
| 121_at       | 2.305589        | 2.383355        | 1            |
| ...          | ...             | ...             | ...          |
| 1553614_a_at | 2.342970        | 2.371501        | 1            |

First, I created binary indicators for all numeric features to flag the presence of outliers in each (sing the Interquartile Range (IQR) method).
After filtering the dataset to focus only on features where outliers were detected (has_outliers=1), I applied a logarithmic transformation to these variables.
I used logarithmic transformation since it compresses large values while preserving the overall trends in order to 
making the data more normally distributed, which is useful for many machine learning models.


## Top Genes Identified in Feature Selection

| Gene        | Score     | P-value  | Rank |
|-------------|-----------|----------|------|
| 210715_s_at | 18.899408 | 0.000080 | 1    |
| 222841_s_at | 17.264344 | 0.000147 | 2    |
| 206284_x_at | 15.158484 | 0.000332 | 3    |
| 201698_s_at | 14.528832 | 0.000426 | 4    |
| 226888_at   | 14.321594 | 0.000462 | 5    |
| 216883_x_at | 13.969812 | 0.000532 | 6    |
| 1555656_at  | 13.771564 | 0.000577 | 7    |
| 235705_at   | 13.770740 | 0.000577 | 8    |
| 1552607_at  | 13.212726 | 0.000724 | 9    |
| 202024_at   | 12.717080 | 0.000888 | 10   |

I used the ANOVA F-test to find genes that show meaningful differences between patients who responded to treatment and those who didn't.
When a gene has a high F-score, it means the expression levels are more different between the two response groups than they are within each group. 
The most meaningful gene is '210715_s_at' with highly significant P-Value (0.000080)


## Model Performance and Interpretation

### Model Comparison
Examing two classification models that might be fit to our use case: Random Forest vs XGBoost

| Model         | Accuracy | Precision | Recall | F1 Score | AUC   |
|---------------|----------|-----------|--------|----------|-------|
| Random Forest | 0.900    | 0.800     | 1.000  | 0.889    | 0.902 |
| XGBoost       | 0.900    | 1.000     | 0.750  | 0.857    | 0.914 |

In our specific use case, I think Random Forest appears to be the better model becouse of the following reasons:

1. Perfect Recall
Random Forest ensures that all true responders will be identified, making it critical when we dealing with real-world clinical scenarios where a false negative (denying treatment to a responder) can be more harmful than a false positive (administering treatment to a non-responder).

2.Lower Precision
Although Random Forest has a lower precision compared to XGBoost, its recall score compensates for this, meaning it will never miss a responder, which is typically more important in clinical settings than avoiding false positives.


3.F1 Score
The F1 score of 0.8889 shows a better balance between precision and recall, which is important in clinical prediction, where both false positives and false negatives have significant consequences.


4.ROC-AUC
While XGBoost has a slightly higher ROC-AUC, this is less relevant given that recall is more important in our case. 

Random Forest’s ROC-AUC of 0.9020 still indicates strong model performance, and the focus should remain on its ability to capture all true responders.

Please see summary plots for model performance metrics: 
1. 'plots/model_comparison.png' - Comparison of all metrics between models
2. 'plots/confusion_matrix_Random Forest.png' - Confusion matrix for Random Forest model
3. 'plots/confusion_matrix_XGBoost.png' - Confusion matrix for XGBoost model

### Final Model: Random Forest
- **Cross-validation performance**: Consistent performance across 5-fold CV : [0.96  0.8  1.0  0.85  0.9 ]

The model showed robust performance across 5-fold cross-validation with AUC scores (mean: 0.90). 
This consistency across suggests the model is stable and not overfitting to specific patients.

## Explainability and Feature Importance 
Proceeding with our chosen model: Random Forest

### Random Forest Feature Importances

| Feature       | Value   |
|--------------|----------|
| 235705_at    | 0.084487 |
| 222841_s_at  | 0.073870 |
| 226888_at    | 0.045718 |
| 1555656_at   | 0.044693 |
| 1552607_at   | 0.043634 |
| 206284_x_at  | 0.039152 |
| 201698_s_at  | 0.038386 |
| 216883_x_at  | 0.035182 |
| 210715_s_at  | 0.029961 |
| DAS28        | 0.015734 |
| 202024_at    | 0.009773 |
| Gender_Female| 0.002153 |

Based on this results, I can suggest the following conclusions:

1. Gene expression markers dominate feature importance: The top 9 features are all gene expression.
They have the most significant role in predicting treatment response.

2. Top predictors identified: The genes "235705_at" and "222841_s_at" stand out with higher importance values (0.084 and 0.074 respectively), contributing over 15% of the model's predictive power combined.


### explainability methods - SHAP - for all patients in Test set (plot file name: 'shap_summary.png')

Based on SHAP value plot for the test data, I can mention several conclusions:

1. Most features show a clear separation between high values (red) pushing predictions toward treatment response and low values (blue) pushing away from response

2. "1555656_at"  have the widest range of SHAP values, making it particularly impactful

3. Higher DAS28 scores (red) generally increase prediction of response, meaning patients with more severe disease activity may be more likely to respond to treatment

Please see summary plot for this section:
'plots/shap_summary.png' 

### explainability methods - SHAP - for specific pateint in test set

Here I wanted to check which features affects the most on specific one patient

***Most 10 affect features on patient number 3***


| Feature      | shap_Value | shap_Value_abs |
|--------------|------------|----------------|
| 1552607_at   | -1.875881  | 1.875881       |
| 226888_at    | 1.540611   | 1.540611       |
| DAS28        | 1.467060   | 1.467060       |
| 201698_s_at  | -1.419114  | 1.419114       |
| 202024_at    | 1.378889   | 1.378889       |
| 222841_s_at  | 1.192558   | 1.192558       |
| Gender_Female| 0.945905   | 0.945905       |
| 210715_s_at  | 0.858058   | 0.858058       |
| 1555656_at   | -0.433670  | 0.433670       |
| 206284_x_at  | 0.417100   | 0.417100       |

Conclusion :

1. Gene "1552607_at" has the strongest absolute impact (1.88), but its negative value actually pushes the prediction away from treatment response.
Similarly, "201698_s_at" has a strong negative influence (-1.42)

2. "226888_at" (1.54), "DAS28" (1.47), and "202024_at" (1.38) are the strongest positive factors,   there might be strongly support a prediction of treatment response.
