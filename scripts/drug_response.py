
#sudo mount -t drvfs G: /mnt/g

import pandas as pd
import numpy as np
import zipfile
from scipy.stats import skew
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import shap


def print_section_header(header_text):

    """
    Print a section header with asterisk borders that adapt to the length of the text.
    """
    # Calculate length of the border based on header text
    border_length = len(header_text) + 8  # Adding extra chars for "*** " and " ***"
    border = "*" * border_length
    
    print(f"\n{border}")
    print(f"*** {header_text} ***")
    print(f"{border}\n")


def load_data(meta_data_path, gene_expression_zip_path, gene_expression_data_path):

    """
    Load Meta Data and Gene Expression Data to dataframes 
    """

    # Extract the contents
    with zipfile.ZipFile(gene_expression_zip_path, 'r') as zip_ref:
        zip_ref.extractall('../data/')

    # Load metadata
    df_meta_data = pd.read_csv(meta_data_path)

    # Load gene expression data
    df_gene_data = pd.read_csv(gene_expression_data_path)

    print_section_header(f"Meta data")
    print(df_meta_data)

    print_section_header(f"Gene expression data")
    print(df_gene_data)


    return df_meta_data, df_gene_data


def merge_tables(df_meta_data, df_gene_data):

    """
    Merge between Meta Data and Gene Expression Data by 
    """

    # Rename columns for better readability
    df_meta_data.columns = df_meta_data.columns.str.strip()  # Remove extra spaces
    df_meta_data.rename(columns={'Response status': 'Response', 'disease activity score (das28)': 'DAS28'}, inplace=True)

    # Convert Response to categorical
    df_meta_data["Response"] = df_meta_data["Response"].map({"Responder": "Responder", "Non_responder": "Non-Responder"})

    #===========================================================================================
    #=========================Merge Tables- Meta Data & Gene Expression ========================
    #===========================================================================================

    # Transpose gene expression data to have genes as columns
    df_gene_data_t = df_gene_data.set_index('ID_REF').T

    # Reset index to make SampleID a column
    df_gene_data_t = df_gene_data_t.reset_index()
    df_gene_data_t = df_gene_data_t.rename(columns={'index': 'SampleID'})


    # Merge with metadata on SampleID
    df_merged = pd.merge(
        df_meta_data,
        df_gene_data_t,
        on='SampleID',
        how='inner'
    )

    print_section_header(f"Merged data")
    print(df_merged)

    # print datasets shape
    print_section_header(f"*Merged tables shapes")
    print(f"Metadata shape: [{df_meta_data.shape[0]:,}][{df_meta_data.shape[1]:,}]")
    print(f"Gene expression shape: [{df_gene_data.shape[0]:,}][{df_gene_data.shape[1]:,}]")
    print(f"Gene expression Transpose shape: [{df_gene_data_t.shape[0]:,}] [{df_gene_data_t.shape[1]:,}]")
    print(f"Merged data shape: [{df_merged.shape[0]:,}] [{df_merged.shape[1]:,}]")


    return df_merged


def get_eda_plots(df_merged):

    """
    Performs exploratory data analysis.
    It takes the merged dataframe containing patient meta data and gene expression values as input and generates:

    1. data types, descriptive stats
    2. Target variable distribution analysis 
    3. Gender vs Response analysis
    4. DAS28 scores by Response analysis
    5. Gene expression distribution analysis
    6. Correlation matrix visualization

    """

    # I sampled only the first 15 columns for examination to ensure the code runs efficiently

    df_sample_for_eda = df_merged.iloc[:, :15]

    # 1. Basic Dataset Information

    print_section_header(f"Data types")
    print(df_sample_for_eda.dtypes)

    print_section_header(f"Descriptive statistics")
    print(df_sample_for_eda.describe().T)


    # 2. Target Variable Analysis

    response_counts = df_merged['Response'].value_counts(dropna=False)
    print_section_header(f"Response distribution")
    print(response_counts)


    sns.countplot(x='Response', data=df_merged, hue='Response', palette='viridis', legend=False)

    plt.title('Distribution of Response', fontsize=15)
    plt.xlabel('Response', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig('../plots/1_1_response_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


    # 3. Feature Analysis

    # 3.1 Gender vs Response analysis

    gender_counts = df_merged['Gender'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Gender', hue='Response', data=df_merged, palette='viridis',  legend=True)

    plt.title('Response Distribution by Gender', fontsize=15)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Response')
    plt.legend(title='Response', loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.savefig('../plots/1_2_gender_response.png', dpi=300, bbox_inches='tight')
    plt.close()


    # 3.2 DAS28 by Response

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Response', y='DAS28', hue='Response', data=df_merged, palette='viridis', legend=False)

    plt.title('DAS28 Scores by Response', fontsize=15)
    plt.xlabel('Response', fontsize=12)
    plt.ylabel('DAS28 Score', fontsize=12)
    plt.savefig('../plots/1_3_das28_by_response.png', dpi=300, bbox_inches='tight')
    plt.close()


    # 4. Gene Expression Analysis

    # I randomly sampled 4 genes to examine their expression distribution

    gene_columns = ['1007_s_at', '1053_at', '117_at', '121_at']

    # 4.1 Distribution of gene expression
    plt.figure(figsize=(16, 12))
    for i, gene in enumerate(gene_columns, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df_merged[gene], kde=True, bins=10)
        plt.title(f'Distribution of {gene}', fontsize=12)
        plt.xlabel('Expression Level', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
    plt.tight_layout()
    plt.savefig('../plots/1_4_gene_expressions_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


    # 5.1 Correlation matrix
    numeric_df = df_sample_for_eda.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_df.corr()

    print_section_header(f"Correlation Matrix*")
    print(correlation_matrix)

    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                mask=mask, vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Numeric Features', fontsize=15)
    plt.tight_layout()
    plt.savefig('../plots/1_5_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def handle_missing_value(df_merged):

    """
    Identifies missing data in the dataset, displaying only columns containing missing values in descending order of frequency. 
    It additionally examines the distribution shape of the DAS28 
    """

    # # Calculate missing values and percentage
    missing_values = df_merged.isnull().sum()
    missing_percentage = (missing_values / len(df_merged)) * 100

    # Filter out columns with no missing values
    df_missing = pd.DataFrame({'Missing Values': missing_values, 'Percentage (%)': np.round(missing_percentage,2)})
    df_missing = df_missing[df_missing['Missing Values'] > 0]

    # Sort from highest to lowest missing values
    df_missing = df_missing.sort_values(by='Missing Values', ascending=False)

    print_section_header(f"Missing values")
    print(df_missing)

    # check DAS28 variable

    das28_skewness = skew(df_merged['DAS28'].dropna())  # Drop NaN values before calculating skewness
    print_section_header(f"Skewness of DAS28")
    print(np.round(das28_skewness,2))


def handle_outliers(df_merged):

    """
    I sampled only 1000 columns for examination to ensure the code runs efficiently
    Identifies outliers in numeric columns using the Interquartile Range (IQR) method, and calculate threshold boundaries.
    I also applying log transformation to columns identify with outlier
    """

    # Get a list of all numeric columns, I took a sample of 1000 numeric column just for example of dealing with ouliers
    numeric_columns = df_merged.select_dtypes(include=['number']).columns.tolist()[0:1000]

    # Create empty lists to store the results
    columns = []
    lower_bounds = []
    upper_bounds = []
    has_outliers = []
    outlier_values = []

    # Calculate outlier ranges for each column
    for column in numeric_columns:
        # Calculate Q1, Q3 and IQR
        Q1 = df_merged[column].quantile(0.25)
        Q3 = df_merged[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers
        outliers = df_merged[(df_merged[column] < lower_bound) | (df_merged[column] > upper_bound)][column].tolist()
        
        # Store the results
        columns.append(column)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
        has_outliers.append(1 if len(outliers) > 0 else 0)

    # Create the outliers threshold DataFrame
    df_outliers = pd.DataFrame({
        'Feature': columns,
        'lower_threshold': lower_bounds,
        'upper_threshold': upper_bounds,
        'has_outliers': has_outliers,
    })


    print_section_header(f"Summerize columns with outlies indication")
    print(df_outliers)

    print_section_header(f"Numeric columns outlier distribution of 1000 Samples")
    print(pd.DataFrame(df_outliers['has_outliers'].value_counts()))

    # Get relevant numerc columns with outliers
    column_with_outliers_lst = list(df_outliers[df_outliers['has_outliers']==1]['Feature'])


    # Apply log transformation to treat outliers
    # Using np.log1p() which calculates log(1+x) to handle potential zeros
    df_merged[column_with_outliers_lst] = np.log1p(df_merged[column_with_outliers_lst])
    df_merged = df_merged.copy() 

    return df_merged


def preprocess_data(df_merged):

    """
    Transforms categorical variables into numerical format by converting the Response status to binary values and Gender to indicator variables, 
    Then removes any rows without valid target.
    """

    df_merged['Response_Binary'] = df_merged['Response'].apply(lambda x: 1 if x == 'Responder' else 0 if x =='Non-Responder' else -1)

    df_merged['Gender_Female'] = df_merged['Gender'].map({'Male': 0, 'Female': 1})

    # remove rows without target label
    cond1 = df_merged['Response_Binary'].isin([0,1])
    df_merged = df_merged[cond1].reset_index(drop=True)

    return df_merged


def feature_selection(df_merged,meta_data_cols,k):

    """
    This function identifies the most significant genes related to treatment response by:

    1. Applying ANOVA F-test to score genes based on their relationship with response
    2. Creating a ranked list of genes with their statistical significance
    3. Returning the top K most important genes for model development

    """

    target_value = 'Response_Binary'

    cols_to_drop = meta_data_cols + [target_value]

    # Prepare feature matrix X (gene expression) and target vector y (response)
    X = df_merged.drop(cols_to_drop,axis=1).values
    y = df_merged[target_value].values


    # Apply feature selection using ANOVA F-test
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)

    # Get scores and p-values
    scores = selector.scores_
    p_values = selector.pvalues_

    # Create DataFrame with gene names, scores and p-values
    # gene_names = merged_data.columns[8:]
    gene_names = list(df_merged.drop(cols_to_drop,axis=1).columns)
    feature_scores = pd.DataFrame({
        'Gene': gene_names,
        'Score': scores,
        'P-value': p_values
    })


    # Sort by score in descending order
    important_genes = feature_scores.sort_values('Score', ascending=False).head(k).reset_index(drop=True)
    important_genes['Rank'] = range(1, k+1)

    print_section_header(f"2. Feature Selection - Identify top {k} genes")
    print(important_genes)

    return important_genes

def plot_model_metrics(df_comparison):

    """
    visualizes ML model performance comparisons metric plots (Accuracy, Precision, Recall, F1 Score, and AUC)   
    """

   
    # Reshaping the metrics table
    plot_df = pd.melt(
                        df_comparison,
                        id_vars=['Model'], 
                        value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],  
                        var_name='Metric',  
                        value_name='Score'  
                    )
    
   

        # Convert Score to numeric explicitly
    plot_df["Score"] = pd.to_numeric(plot_df["Score"])

    # Reset all previous plots
    plt.close('all')

    # Create a simple bar plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette="viridis")

    # Set y-axis to start at 0
    plt.ylim(0, 1.1)

    # Add labels and title
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Score', fontsize=14)

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')

    plt.tight_layout()
    plt.savefig('../plots/model_comparison.png', dpi=300, bbox_inches='tight')


def evaluate_model(model_name, model, X, y, X_train, X_test, y_train, y_test):

    """
    Creating a preprocessing and classification pipeline that handles missing values and feature scaling.
    It trains the model, generates predictions, and calculates multiple performance metrics including accuracy, precision, recall, F1 score, and AUC. 
    The function also implements cross-validation for robust evaluation, plot the confusion matrix, and returns both the trained pipeline and a formatted results dataframe for model comparison.
    """

    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('mean_imputer', SimpleImputer(strategy='mean'), X.columns), #impute missing values by mean
            ('scaler', StandardScaler(), X.columns) #scales each feature (column) to have a mean of 0 and a standard deviation of 1.
        ]
    )

    # Build pipeline
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Get probability for class 1
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']
    precision = classification_report(y_test, y_pred, output_dict=True)['1']['precision']
    recall = classification_report(y_test, y_pred, output_dict=True)['1']['recall']

    # Perform cross-validation
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')  # Using ROC-AUC as evaluation metric
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative (0)', 'Positive (1)'],
                yticklabels=['Negative (0)', 'Positive (1)'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=15)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'../plots/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create DataFrame for results
    results_df = pd.DataFrame({
        "Model": [model_name],
        "Accuracy": [f"{accuracy:.3f}"],
        "Precision": [f"{precision:.3f}"],
        "Recall": [f"{recall:.3f}"],
        "F1 Score": [f"{f1:.3f}"],
        "AUC": [f"{np.mean(scores):.3f}"]
    })

    print_section_header(f"Cross validation scores for {model_name}") 
    print(scores)

    return pipeline, results_df


def feature_importance(X,model):

    """
    The  function extracts and ranks the importance of each feature (gene) from a trained model pipeline.
    This helps identify the features most relevant to treatment response prediction.
    """

    model_importances = model.named_steps['classifier'].feature_importances_

    rf_importance_dct = {}

    #Extract importance values for column is in the model
    for feature, importance in zip(X.columns, model_importances):
        rf_importance_dct[feature]=importance

    model_importances_df = pd.DataFrame(list(rf_importance_dct.items()), columns=['Feature', 'Value']).sort_values(by='Value',ascending=False).reset_index(drop=True)

    print_section_header("4.1 Random Forest Feature Importances")
    print(model_importances_df)


def model_shap_explanatory(X_test, pipeline):

    """
    The function analyzes and visualizes how different features influence model predictions using SHAP values. 
    It extracts the classifier from the pipeline, transforms the test data through the preprocessing steps, then generates SHAP explanations.
    The function creates two visualizations: 
        1. A summary plot showing the overall feature importance across all samples.
        2. Detailed analysis of feature impacts for a specific patient (patient #3). 
    
    These visualizations help identify which gene expressions most strongly influence the model's prediction of treatment response.

    """

    feature_names = pipeline.feature_names_in_

    # Now, we explain the model using SHAP , Get the Random Forest model from the pipeline
    model = pipeline.named_steps['classifier']

    total_pipeline_steps = len(pipeline.steps)-1

    preprocessor = pipeline.named_steps['preprocessing']

    # I used SHAP to explain the model
    explainer = shap.TreeExplainer(model)

    # Transform the test data using the preprocessing pipeline
    X_test_transformed = preprocessor.transformers_[total_pipeline_steps][1].transform(X_test)

    # Convert transformed test data to DataFrame
    X_test_df_shap = pd.DataFrame(X_test_transformed, columns=feature_names)

    # Load SHAP values DataFrame
    shap_values = X_test_df_shap.values
    feature_names = X_test_df_shap.columns


    #============================
    #=====SHAP Summary Plot======
    #============================

    plt.figure()
    shap.summary_plot(shap_values, features=X_test_df_shap, feature_names=feature_names, show=False)

    # Save the plot as an image
    plt.savefig("../plots/shap_summary.png", dpi=300, bbox_inches="tight")  # Save as PNG (or change to .jpg/.pdf)
    plt.close()

    #===================================================
    #=====Analyze SHAP Values for specific pateint======
    #===================================================

    # Lets take a look what feature affect the most on patient number 3

    # Extract SHAP values for pateint 3 (assuming the first row represents pateint 3)
    user_shap = X_test_df_shap.iloc[3, :]  # Get the first row


    # Sort features by absolute SHAP values (most impactful first)
    user_shap_sorted = pd.DataFrame({
        'Feature': feature_names,   # Feature names
        'shap_Value': user_shap,      #  SHAP values
        'shap_Value_abs': user_shap.abs()      # Absolute SHAP values

    }).sort_values(by="shap_Value_abs", ascending=False).reset_index(drop=True)

    # Display top features affecting pateint 3

    print_section_header("Most 10 affect features on patient number 3")
    print(user_shap_sorted.head(10))  # Show the top 10 most impactful features

def main():

    """
    The main function orchestrates the entire drug response prediction workflow:

    1. Loads and merges metadata and gene expression datasets
    2. Performs exploratory data analysis with visualizations
    3. Handles missing values and outliers
    4. Preprocesses data by encoding categorical variables
    5. Selects the most important genes using statistical methods
    6. Builds and evaluates two machine learning models (Random Forest and XGBoost)
    7. Compares model performance metrics with visualizations
    8. Analyzes feature importance to identify key predictive genes
    9. Uses SHAP values to explain model predictions and provide insights into treatment response mechanisms

    """

    #=======================================================
    #=====Load data files - Meta Data & Gene Expression ====
    #=======================================================

    meta_data_path = '../data/meta_data.csv'
    gene_expression_zip_path = '../data/gene_expression.zip' #gene_expression # gene_expression_try
    gene_expression_data_path = '../data/gene_expression.csv'


    # Load meta data and gene expression data
    df_meta_data, df_gene_data = load_data(meta_data_path, gene_expression_zip_path, gene_expression_data_path)

    # Merge between the dataframe
    df_merged = merge_tables(df_meta_data, df_gene_data)

    # Get Datasets columns names
    meta_data_cols = list(df_meta_data.columns)
    gene_data_cols = [col for col in df_merged.columns if col not in meta_data_cols]


    #==============================================
    #====== 1. Exploratory Data Analysis (EDA) ====
    #==============================================

    get_eda_plots(df_merged)

    #===================================
    #===== Handeling missing values ====
    #===================================

    handle_missing_value(df_merged)

    #===================================
    #=== Highlight outliers ============
    #===================================

    df_merged = handle_outliers(df_merged)


    #===================================
    #== Preproccesing data =============
    #===================================

    df_merged = preprocess_data(df_merged)

    #================================================
    #=== 2. Feature selection - Best K  =============
    #================================================

    k = 10

    important_genes = feature_selection(df_merged,meta_data_cols,k)

    # ================================================
    # ==== 3. Predictive Modeling  ===================
    # ================================================

    # Examing two classification models that might be fit to our use case: Random Forest vs XGBoost

    target = 'Response_Binary'

    relevant_meta_data_cols = ['DAS28', 'Gender_Female', f"{target}"]

    selected_features = relevant_meta_data_cols + important_genes['Gene'].tolist() # From Feature selection stage

    # Convert categorical variables to numerical
    model_data = df_merged.copy()

    # Select features and target
    X = model_data[selected_features].drop([target],axis=1)
    y = model_data[target]

        # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')  
    }


    df_comparison = pd.DataFrame()
    pipeline_model = {}

    # Run evaluation for both models
    for model_name, clf in classifiers.items():
        
        # Build pipeline
        pipeline, metric_results = evaluate_model(model_name, clf, X, y, X_train, X_test, y_train, y_test) # save metric results for each model

        pipeline_model[model_name] = pipeline

        df_comparison = pd.concat([df_comparison,metric_results],axis=0)


    df_comparison = df_comparison.reset_index(drop=True)

    print_section_header("3. Comparison models metrics")
    print(df_comparison)

    plot_model_metrics(df_comparison)

    # =================================================
    # ==== 4. Explainability and Feature Importance ===
    # =================================================

    # Proceed the next step with ouer chosen model: Random Forest

    #======================================
    #== 4.1 RF feature importance =========
    #======================================

    rf_model = pipeline_model['Random Forest']

    feature_importance(X,rf_model)

    #===========================================
    #== 4.2 model explanation with SHAP ========
    #===========================================

    model_shap_explanatory(X_test, rf_model)


if __name__ == "__main__":
    main()