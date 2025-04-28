# Missing Values Analysis and Handling Script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def analyze_missing_values(df):
    """
    Analyze missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with missing value statistics
    """
    # Calculate missing values count and percentage
    missing = pd.DataFrame({
        'Missing Count': df.isnull().sum(),
        'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('Missing Percentage', ascending=False)
    
    # Add data types
    missing['Data Type'] = [df[col].dtype for col in missing.index]
    
    # Only show columns with missing values
    missing = missing[missing['Missing Count'] > 0]
    
    print(f"Total features with missing values: {len(missing)}")
    return missing


def get_columns_with_missing_values(df):
    """
    Return a list of column names that have missing values in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        list: List of column names that have missing values
    """
    # Get columns with at least one missing value
    missing_columns = df.columns[df.isnull().any()].tolist()
    
    print(f"Found {len(missing_columns)} columns with missing values")
    return missing_columns


def visualize_missing_values(df):
    """
    Visualize missing values in multiple ways.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    # 1. Matrix plot to visualize missing values pattern
    plt.figure(figsize=(12, 6))
    msno.matrix(df)
    plt.title('Missing Value Patterns')
    plt.tight_layout()
    plt.show()
    
    
    # 2. Correlation of missingness
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]  # Only show columns with missing values

    if len(missing_pct) > 1:
        plt.figure(figsize=(10, 8))
        msno.heatmap(df)
        plt.title('Correlation of Missingness')
        plt.tight_layout()
        plt.show()


def check_missing_at_random(df, missing_cols):
    """
    Check if data is missing at random by analyzing relationships between missing values and other columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        missing_cols (list): List of columns with missing values
        
    Returns:
        dict: Evidence of Missing At Random (MAR) relationships
    """
    mar_evidence = {}
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    print("\n=== ANALYZING MISSING VALUE PATTERNS ===")
    
    for col in missing_cols:
        missing_indicator = df[col].isna()
        missing_count = missing_indicator.sum()
        
        if missing_count == 0:
            continue
            
        mar_evidence[col] = {}
        print(f"\nAnalyzing missingness pattern for '{col}' ({missing_count} missing values):")
        
        # Check if values of numeric columns differ by missingness
        for num_col in numeric_cols:
            if num_col != col and df[num_col].notna().any():
                present_mean = df.loc[~missing_indicator, num_col].mean()
                missing_mean = df.loc[missing_indicator, num_col].mean()
                
                # Calculate percent difference
                diff_pct = abs((present_mean - missing_mean) / present_mean * 100) if present_mean != 0 else 0
                
                if diff_pct > 10:  # Arbitrary threshold for significant difference
                    print(f"  - '{num_col}' shows different values when '{col}' is missing:")
                    print(f"    When '{col}' is present: mean = {present_mean:.2f}")
                    print(f"    When '{col}' is missing: mean = {missing_mean:.2f}")
                    print(f"    Percentage difference: {diff_pct:.2f}%")
                    
                    mar_evidence[col][num_col] = {
                        'present_mean': present_mean,
                        'missing_mean': missing_mean,
                        'diff_percentage': diff_pct
                    }
        
        # Check if distribution of categorical columns differs by missingness
        for cat_col in categorical_cols:
            if cat_col != col and df[cat_col].notna().any():
                # Count occurrences for each category when target is present vs missing
                present_counts = df.loc[~missing_indicator, cat_col].value_counts(normalize=True)
                missing_counts = df.loc[missing_indicator, cat_col].value_counts(normalize=True)
                
                # Compare distributions for common categories
                common_cats = set(present_counts.index) & set(missing_counts.index)
                
                if common_cats:
                    max_diff = max(abs(present_counts.get(cat, 0) - missing_counts.get(cat, 0)) for cat in common_cats)
                    
                    if max_diff > 0.1:  # Arbitrary threshold for significant difference
                        print(f"  - '{cat_col}' shows different distribution when '{col}' is missing")
                        mar_evidence[col][cat_col] = {'max_distribution_diff': max_diff}
    
    return mar_evidence



def impute_numeric_features(df, columns, feature_strategies, default_method='median'):
    """
    Impute missing values in numeric features using different strategies per feature.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of numeric columns to impute
        feature_strategies (dict): Dictionary mapping column names to specific imputation methods
        default_method (str): Default imputation method if no specific strategy is provided
        
    Returns:
        pd.DataFrame: Dataframe with imputed values
    """
    df_imputed = df.copy()
    feature_strategies = feature_strategies or {}
    
    # Process remaining columns with their specific strategies
    cols_by_strategy = {}

    # Group columns by their imputation strategy
    for col in columns:
        if col not in df_imputed.columns:
            continue  # Skip columns that don't exist
            
        strategy = feature_strategies.get(col, default_method)
        if strategy not in cols_by_strategy:
            cols_by_strategy[strategy] = []
        cols_by_strategy[strategy].append(col)
    
    # Apply imputation for each strategy group
    for strategy, strategy_cols in cols_by_strategy.items():
        if not strategy_cols:
            continue
            
        # Check if these columns actually have missing values
        missing_counts = {col: df_imputed[col].isna().sum() for col in strategy_cols}
        cols_with_missing = [col for col, count in missing_counts.items() if count > 0]
        
        if not cols_with_missing:
            print(f"No missing values found in columns {strategy_cols}")
            continue
            
        print(f"Imputing columns {cols_with_missing} using '{strategy}' strategy")
        
        if strategy == 'mean':
            for col in cols_with_missing:
                mean_value = df_imputed[col].mean()
                print(f"  Filling {col} missing values with mean: {mean_value:.4f}")
                df_imputed[col] = df_imputed[col].fillna(mean_value)
            
        elif strategy == 'median':
            for col in cols_with_missing:
                median_value = df_imputed[col].median()
                print(f"  Filling {col} missing values with median: {median_value:.4f}")
                df_imputed[col] = df_imputed[col].fillna(median_value)
            
        elif strategy == 'mode':
            for col in cols_with_missing:
                mode_value = df_imputed[col].mode()[0]
                print(f"  Filling {col} missing values with mode: {mode_value}")
                df_imputed[col] = df_imputed[col].fillna(mode_value)
            
        elif strategy == 'constant':
            for col in cols_with_missing:
                print(f"  Filling {col} missing values with constant: 0")
                df_imputed[col] = df_imputed[col].fillna(0)
            
        elif strategy == 'mice':
            print(f"  Applying MICE imputation to {cols_with_missing}")
            numeric_cols = df_imputed.select_dtypes(include=['number']).columns.tolist()
            numeric_df = df_imputed[numeric_cols].copy()
            
            # Initial simple imputation for other features to make MICE work
            for nc in numeric_cols:
                if nc not in cols_with_missing and numeric_df[nc].isna().any():
                    numeric_df[nc] = numeric_df[nc].fillna(numeric_df[nc].median())
            
            # Apply MICE imputation
            imputer = IterativeImputer(random_state=42, max_iter=10)
            numeric_df_imputed = pd.DataFrame(
                imputer.fit_transform(numeric_df),
                columns=numeric_cols
            )
            
            # Copy back only the columns we wanted to impute
            for col in cols_with_missing:
                df_imputed[col] = numeric_df_imputed[col].values  # Use .values to ensure proper assignment
        
        else:
            raise ValueError(f"Unknown imputation method: {strategy}")
        
        # Verify imputation worked for this strategy
        remaining_missing = sum(df_imputed[col].isna().sum() for col in cols_with_missing)
        print(f"  Remaining missing values after {strategy} imputation: {remaining_missing}")
    
    return df_imputed


def impute_categorical_features(df, columns, feature_strategies, default_method='mode'):
    """
    Impute missing values in categorical features using different strategies per feature.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of categorical columns to impute
        feature_strategies (dict): Dictionary mapping column names to specific imputation methods
        default_method (str): Default imputation method if no specific strategy is provided
        
    Returns:
        pd.DataFrame: Dataframe with imputed values
    """
    df_imputed = df.copy()
    feature_strategies = feature_strategies or {}
    
    # Group columns by their imputation strategy
    cols_by_strategy = {}
    for col in columns:
        if col not in df_imputed.columns:
            continue  # Skip columns that don't exist
            
        strategy = feature_strategies.get(col, default_method)
        if strategy not in cols_by_strategy:
            cols_by_strategy[strategy] = []
        cols_by_strategy[strategy].append(col)
    
    # Apply imputation for each strategy group
    for strategy, strategy_cols in cols_by_strategy.items():
        if not strategy_cols:
            continue
            
        # Check if these columns actually have missing values
        missing_counts = {col: df_imputed[col].isna().sum() for col in strategy_cols}
        cols_with_missing = [col for col, count in missing_counts.items() if count > 0]
        
        if not cols_with_missing:
            print(f"No missing values found in columns {strategy_cols}")
            continue
            
        print(f"Imputing columns {cols_with_missing} using '{strategy}' strategy")
        
        if strategy == 'mode':
            for col in cols_with_missing:
                mode_value = df_imputed[col].mode()[0]
                print(f"  Filling {col} missing values with mode: {mode_value}")
                df_imputed[col] = df_imputed[col].fillna(mode_value)
        
        elif strategy == 'constant':
            for col in cols_with_missing:
                print(f"  Filling {col} missing values with constant: 'Unknown'")
                df_imputed[col] = df_imputed[col].fillna('Unknown')
        
        elif strategy == 'new_category':
            for col in cols_with_missing:
                print(f"  Filling {col} missing values with new category: 'Missing'")
                df_imputed[col] = df_imputed[col].fillna('Missing')
        
        else:
            raise ValueError(f"Unknown imputation method: {strategy}")
        
        # Verify imputation worked for this strategy group
        remaining_missing = sum(df_imputed[col].isna().sum() for col in cols_with_missing)
        print(f"  Remaining missing values after {strategy} imputation: {remaining_missing}")
    
    return df_imputed




def evaluate_imputation(original_df, imputed_df, numeric_columns, categorical_columns):
    """
    Evaluate the impact of imputation on data distribution.
    
    Args:
        original_df (pd.DataFrame): Original dataframe with missing values
        imputed_df (pd.DataFrame): Imputed dataframe
        numeric_columns (list): List of numeric columns
        categorical_columns (list): List of categorical columns
    """
    print("\n=== IMPUTATION EVALUATION ===")
    
    # Evaluate numeric columns
    for col in numeric_columns:
        if col not in original_df.columns or col not in imputed_df.columns:
            continue
            
        if original_df[col].isna().sum() > 0:
            print(f"\nEvaluating imputation for numeric column '{col}':")
            
            # Compare basic statistics
            original_stats = original_df[col].describe()
            imputed_stats = imputed_df[col].describe()
            
            print("  Original vs Imputed Statistics:")
            comparison = pd.DataFrame({
                'Original': original_stats,
                'Imputed': imputed_stats,
                'Difference %': ((imputed_stats - original_stats) / original_stats * 100).round(2)
            })
            print(comparison)
            
            # Visualize distributions
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            sns.histplot(original_df[col].dropna(), kde=True, color='blue', label='Original (non-missing)')
            plt.title(f'Original Distribution (non-missing)\n{col}')
            
            plt.subplot(1, 2, 2)
            sns.histplot(imputed_df[col], kde=True, color='green', label='After Imputation')
            plt.title(f'Distribution After Imputation\n{col}')
            
            plt.tight_layout()
            plt.show()
    
    # Evaluate categorical columns
    for col in categorical_columns:
        if col not in original_df.columns or col not in imputed_df.columns:
            continue
            
        if original_df[col].isna().sum() > 0:
            print(f"\nEvaluating imputation for categorical column '{col}':")
            
            # Compare value counts
            original_counts = original_df[col].value_counts(dropna=False, normalize=True).head(10)
            imputed_counts = imputed_df[col].value_counts(dropna=False, normalize=True).head(10)
            
            print("  Original vs Imputed Top Categories (%):")
            print(pd.DataFrame({
                'Original %': original_counts * 100,
                'Imputed %': imputed_counts * 100
            }))
            
            # Visualize top categories
            plt.figure(figsize=(14, 6))
            
            plt.subplot(1, 2, 1)
            original_df[col].value_counts(dropna=False).head(10).plot(kind='bar', color='blue')
            plt.title(f'Original Top Categories\n{col}')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 2, 2)
            imputed_df[col].value_counts().head(10).plot(kind='bar', color='green')
            plt.title(f'Top Categories After Imputation\n{col}')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()

def handle_missing_values(df, numeric_strategy='median', categorical_strategy='mode', 
                          custom_numeric_values=None, custom_categorical_values=None):
    """
    Complete pipeline for handling missing values in a dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_strategy (str): Strategy for imputing numeric values
        categorical_strategy (str): Strategy for imputing categorical values
        custom_numeric_values (dict): Custom imputation values for numeric columns
        custom_categorical_values (dict): Custom imputation values for categorical columns
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    print("=== MISSING VALUES ANALYSIS AND HANDLING ===")
    print(f"Dataset shape: {df.shape}")
    
    # 1. Analyze missing values
    missing_stats = analyze_missing_values(df)
    print("\nMissing Value Statistics:")
    print(missing_stats)
    
    # 2. Visualize missing values
    visualize_missing_values(df)
    
    # 3. Check if missing at random (MAR)
    missing_cols = missing_stats.index.tolist()
    mar_evidence = check_missing_at_random(df, missing_cols)
    
    # 4. Split columns by data type
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Filter to only include columns with missing values
    numeric_missing = [col for col in numeric_cols if col in missing_cols]
    categorical_missing = [col for col in categorical_cols if col in missing_cols]
    
    print(f"\nNumeric columns with missing values ({len(numeric_missing)}): {numeric_missing}")
    print(f"Categorical columns with missing values ({len(categorical_missing)}): {categorical_missing}")
    
    # 5. Impute missing values
    print("\n=== IMPUTING MISSING VALUES ===")
    
    # Impute numeric features
    df_imputed = impute_numeric_features(
        df, 
        numeric_missing, 
        method=numeric_strategy,
        custom_values=custom_numeric_values
    )
    
    # Impute categorical features
    df_imputed = impute_categorical_features(
        df_imputed, 
        categorical_missing, 
        method=categorical_strategy,
        custom_values=custom_categorical_values
    )
    
    # 6. Evaluate imputation
    evaluate_imputation(df, df_imputed, numeric_missing, categorical_missing)
    
    # 7. Final report
    print("\n=== MISSING VALUES HANDLING COMPLETE ===")
    print(f"Original missing values: {df.isnull().sum().sum()}")
    print(f"Remaining missing values: {df_imputed.isnull().sum().sum()}")
    
    return df_imputed

# Example usage
if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv("data/credit_score_dataset.csv")
    
    # Define custom imputation values (optional)
    custom_numeric_values = {
        'Annual_Income': df['Annual_Income'].median(),  # Use median for income
        'Monthly_Inhand_Salary': df['Monthly_Inhand_Salary'].median(),  # Use median for salary
        'Num_Credit_Inquiries': 0  # Use 0 for number of credit inquiries
    }
    
    custom_categorical_values = {
        'Occupation': 'Other',  # Use 'Other' for missing occupations
        'Type_of_Loan': 'Unknown'  # Use 'Unknown' for missing loan types
    }
    
    # Handle missing values with the complete pipeline
    df_clean = handle_missing_values(
        df,
        numeric_strategy='median',
        categorical_strategy='mode',
        custom_numeric_values=custom_numeric_values,
        custom_categorical_values=custom_categorical_values
    )
    
    # Save cleaned dataset
    df_clean.to_csv("data/clean_credit_score_dataset.csv", index=False)