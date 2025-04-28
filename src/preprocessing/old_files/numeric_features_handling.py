# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
        


def check_problematic_values(df, columns):
    """
    Check and display problematic non-numeric values in specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of column names to check
        
    Returns:
        dict: Dictionary with problematic values for each column
    """
    problematic_values = {}
    
    for col in columns:
        # Get mask of non-numeric values (excluding NaN)
        non_numeric_mask = pd.to_numeric(df[col], errors='coerce').isna() & df[col].notna()
        
        if non_numeric_mask.any():
            values = df[col][non_numeric_mask].value_counts()
            problematic_values[col] = values
            
            print(f"\nProblematic values in column '{col}':")
            print(values)
            print(f"Total problematic entries: {len(df[col][non_numeric_mask])}")
    
    return problematic_values


def clean_numeric_values(df, column, cleaning_rules=None):
    """
    Clean numeric values based on specified rules.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to clean
        cleaning_rules (dict, optional): Dictionary of cleaning rules
            Example: {
                'remove_suffix': '_',
                'remove_special_chars': ['$', '%']
            }
    
    Returns:
        pd.Series: Cleaned numeric column
    """
    df_copy = df.copy()
    series = df_copy[column].copy()
    
    if cleaning_rules is None:
        print("No cleaning rules provided")
        return series

    if isinstance(series, pd.Series):
        # Apply cleaning rules
        if cleaning_rules.get('remove_suffix'):
            series = series.astype(str).str.rstrip(cleaning_rules['remove_suffix'])
            
        if cleaning_rules.get('remove_special_chars'):
            for char in cleaning_rules['remove_special_chars']:
                series = series.astype(str).str.replace(char, '')
        
        # Convert to numeric
        series = pd.to_numeric(series, errors='coerce')
    
    return series



def analyze_numeric_feature(df, column_name, mode='Before Cleaning'):
    """
    Analyze a single feature/column, showing key statistics and visualizations
    before and after cleaning
    """
    print(f"\n{'-'*50}")
    print(f"Analysis of {column_name}")
    print(f"{'-'*50}")
    

    # Before or After cleaning
    print(f"\n{mode.upper()}:")
    print(f"Missing values: {df[column_name].isnull().sum()} ({(df[column_name].isnull().sum()/len(df))*100:.2f}%)")
    
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    
    # Statistics
    print("\nStatistics:")
    print(df[column_name].describe())
    
    # Distribution plot
    sns.histplot(df[column_name].dropna(), kde=True)
    plt.title(f"Distribution of {column_name}\n{mode}")
    plt.xticks(rotation=45)
    
    # Box plot for outliers
    plt.subplot(122)
    sns.boxplot(y=df[column_name].dropna())
    plt.title(f"Boxplot of {column_name}\n{mode}")
        
    plt.tight_layout()
    plt.show()


def handle_outliers_iqr(df, column, multiplier=1.5):
    """Replace outliers based on IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    if outlier_mask.any():
        print(f"  Found {outlier_mask.sum()} outliers using IQR method")
        print(f"  IQR bounds: {lower_bound:.2f} to {upper_bound:.2f}")
        df.loc[outlier_mask, column] = np.nan
    
    return df

def handle_outliers_zscore(df, column, threshold=3):
    """Replace outliers based on Z-score method"""
    mean = df[column].mean()
    std = df[column].std()
    
    # Avoid division by zero
    if std == 0:
        return df, pd.Series([False] * len(df))
    
    z_scores = np.abs((df[column] - mean) / std)
    outlier_mask = z_scores > threshold
    
    if outlier_mask.any():
        print(f"  Found {outlier_mask.sum()} outliers using Z-score method")
        print(f"  Z-score threshold: {threshold} (mean: {mean:.2f}, std: {std:.2f})")
        df.loc[outlier_mask, column] = np.nan
    
    return df


def handle_inconsistent_numeric_values(df, numeric_feature, constraints):
    """
    Handle inconsistent numeric values in a numeric feature by:
    1. Negative values when inappropriate
    2. Decimal values when integers expected
    3. Outliers using appropriate methods
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_feature (str): Name of the numeric column to process
        constraints (dict): Dictionary of constraints for each feature
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Validate constraints dictionary first - fail fast
    required_keys = ['min', 'max', 'integer', 'outlier_method']
    
    # Check if feature has constraints defined
    if numeric_feature not in constraints:
        raise ValueError(f"No constraints defined for column '{numeric_feature}'")
            
    # Check if all required keys exist for this column
    missing_keys = []
    for key in required_keys:
        if key not in constraints[numeric_feature]:
            missing_keys.append(key)
    
    # Raise error if any constraints are missing
    if missing_keys:
        error_msg = f"Missing constraints for column '{numeric_feature}': {', '.join(missing_keys)}"
        raise ValueError(error_msg)
    
    # If we get here, all constraints are properly defined
    df_clean = df.copy()
    
    print(f"\nCleaning {numeric_feature}...")
    
    # 1. Handle negative/below-minimum values
    if constraints[numeric_feature]['min'] is not None:
        neg_mask = df_clean[numeric_feature] < constraints[numeric_feature]['min']
        if neg_mask.any():
            print(f"  Found {neg_mask.sum()} below-minimum constraint. Replacing with NaN")
            df_clean.loc[neg_mask, numeric_feature] = np.nan
    
    # 2. Handle decimal values for integer features
    if constraints[numeric_feature]['integer']:
        decimal_mask = df_clean[numeric_feature].notna() & (df_clean[numeric_feature] != df_clean[numeric_feature].round())
        if decimal_mask.any():
            print(f"  Found {decimal_mask.sum()} decimal values in integer feature. Rounding to nearest integer")
            df_clean.loc[decimal_mask, numeric_feature] = df_clean.loc[decimal_mask, numeric_feature].round()
            # Convert whole column to integer type
            df_clean[numeric_feature] = df_clean[numeric_feature].astype('Int64')  # Handles NaN values
    
    # 3. Handle outliers based on specified method
    method = constraints[numeric_feature]['outlier_method']
    if method == 'domain':
        if constraints[numeric_feature]['max'] is not None:
            high_mask = df_clean[numeric_feature] > constraints[numeric_feature]['max']
            if high_mask.any():
                print(f"  Found {high_mask.sum()} values above max threshold {constraints[numeric_feature]['max']}")
                df_clean.loc[high_mask, numeric_feature] = np.nan
    
    elif method == 'iqr':
        print(f"  Using IQR method for {numeric_feature}")
        df_clean = handle_outliers_iqr(df_clean, numeric_feature)
        
    elif method == 'zscore':
        print(f"  Using Z-score method for {numeric_feature}")
        df_clean = handle_outliers_zscore(df_clean, numeric_feature)
    
    elif method != 'none':
        raise ValueError(f"Unknown outlier_method '{method}' for column '{numeric_feature}'")
    
    return df_clean




def analyze_missingness_comprehensively(df):
    """Comprehensive analysis of missing data in one function"""
    
    plt.figure(figsize=(10, 6))
    msno.matrix(df)
    plt.title('Missing Value Patterns')
    plt.tight_layout()
    plt.show()
    
    # 2. Check for relationships between missingness and observed values
    missing_cols = df.columns[df.isna().any()].tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Create simplified MAR check
    mar_evidence = {}
    for col in missing_cols:
        missing_indicator = df[col].isna()
        predictors = []
        
        # Check if values of other numeric columns differ by missingness
        for num_col in numeric_cols:
            if num_col != col and df[num_col].notna().any():
                vals_when_present = df.loc[~missing_indicator, num_col].dropna()
                vals_when_missing = df.loc[missing_indicator, num_col].dropna()
                
                if len(vals_when_present) > 10 and len(vals_when_missing) > 10:
                    from scipy.stats import ttest_ind
                    try:
                        _, p_val = ttest_ind(vals_when_present, vals_when_missing)
                        if p_val < 0.05:
                            diff_pct = (vals_when_missing.mean() - vals_when_present.mean()) / vals_when_present.mean() * 100
                            predictors.append((num_col, p_val, diff_pct))
                    except:
                        pass
        
        mar_evidence[col] = sorted(predictors, key=lambda x: x[1])
    
    # 3. Summarize findings
    print("MISSING DATA ANALYSIS SUMMARY")
    print("=" * 40)
    
    print("\nMissing Data Percentages:")
    missing_pct = df.isna().mean() * 100
    for col, pct in missing_pct[missing_pct > 0].sort_values(ascending=False).items():
        print(f"  {col}: {pct:.2f}%")
    
    print("\nPossible Missing Mechanisms:")
    
    for col in missing_cols:
        predictors = mar_evidence[col]
        if predictors:
            print(f"\n  {col} - Likely MAR (Missing At Random):")
            print(f"  Values are more likely to be missing based on these variables:")
            for pred, p_val, diff_pct in predictors[:3]:
                direction = "higher" if diff_pct > 0 else "lower"
                print(f"    * {pred}: {abs(diff_pct):.1f}% {direction} when {col} is missing (p={p_val:.4f})")
            
            print(f"  Recommended approach: Use {', '.join([p[0] for p in predictors[:2]])} as predictors in imputation")
        else:
            print(f"\n  {col} - Possibly MCAR (or insufficient data to determine):")
            print(f"  No clear relationship between missingness and other variables")
            print(f"  Recommended approach: Median imputation or KNN if % missing is small")
    
    print("\nNOTE: MNAR cannot be ruled out without domain knowledge")
    print("Consider the data collection process and potentially missing data mechanism")


from scipy.stats import chi2_contingency, ttest_ind

def enhanced_missing_data_analysis(df):
    """Comprehensive analysis of missing data with enhanced visualizations and tests"""
    
    try:
        from statsmodels.stats.missing import missing_completely_at_random
        can_run_littles_test = True
    except:
        can_run_littles_test = False
    
    print("MISSING DATA ANALYSIS")
    print("=" * 50)
    
    # 1) Basic summary statistics
    missing_counts = df.isnull().sum()
    missing_percent = 100 * missing_counts / len(df)
    
    missing_data = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percent': missing_percent
    }).sort_values('Missing Percent', ascending=False)
    
    print("\nMissing Data Summary (columns with missing values):")
    print(missing_data[missing_data['Missing Count'] > 0])
    
    # 2a) The "barcode" style matrix
    plt.figure(figsize=(12, 4))
    msno.matrix(df, fontsize=12, sparkline=False)
    plt.title("Missing‑Value Matrix")
    plt.show()
    
    # 2b) Heatmap of missingness correlations
    plt.figure(figsize=(10, 8))
    corr = df.isnull().corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Feature–Feature Missingness Correlation")
    plt.show()
    
    # 3) Little's MCAR test (if available)
    if can_run_littles_test:
        print("\nLittle's MCAR Test:")
        try:
            result = missing_completely_at_random(df)
            print(f"Test statistic: {result.statistic:.2f}")
            print(f"p-value: {result.pvalue:.4f}")
            
            if result.pvalue < 0.05:
                mcar_conclusion = "REJECT MCAR hypothesis (data is likely NOT MCAR)"
            else:
                mcar_conclusion = "CANNOT REJECT MCAR hypothesis (data might be MCAR)"
            print(mcar_conclusion)
        except Exception as e:
            print(f"Could not perform Little's MCAR test: {e}")
    
    # 4) MAR testing for numeric variables using t-tests
    missing_cols = df.columns[df.isnull().any()].tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print("\nMAR Testing (relationships between missingness and observed values):")
    
    # 4a) For numeric predictors with numeric missing columns
    mar_evidence_numeric = {}
    for col in missing_cols:
        missing_indicator = df[col].isnull()
        predictors = []
        
        for num_col in numeric_cols:
            if num_col != col and df[num_col].notna().any():
                vals_when_present = df.loc[~missing_indicator, num_col].dropna()
                vals_when_missing = df.loc[missing_indicator, num_col].dropna()
                
                if len(vals_when_present) > 10 and len(vals_when_missing) > 10:
                    try:
                        _, p_val = ttest_ind(vals_when_present, vals_when_missing)
                        if p_val < 0.05:
                            diff_pct = (vals_when_missing.mean() - vals_when_present.mean()) / vals_when_present.mean() * 100
                            predictors.append((num_col, p_val, diff_pct))
                    except:
                        pass
        
        mar_evidence_numeric[col] = sorted(predictors, key=lambda x: x[1])
    
    # 4b) For categorical predictors with numeric missing columns
    mar_evidence_categorical = {}
    for col in missing_cols:
        missing_indicator = df[col].isnull()
        predictors = []
        
        for cat_col in categorical_cols:
            if cat_col != col and df[cat_col].notna().any():
                # Create contingency table: cat_col vs missingness
                try:
                    contingency = pd.crosstab(
                        df[cat_col].fillna('_MISSING_'), 
                        missing_indicator
                    )
                    
                    if contingency.shape[0] > 1 and contingency.shape[1] == 2:
                        chi2, p_val, _, _ = chi2_contingency(contingency)
                        if p_val < 0.05:
                            # Calculate effect size (Cramer's V)
                            n = contingency.sum().sum()
                            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                            predictors.append((cat_col, p_val, cramers_v))
                except:
                    pass
        
        mar_evidence_categorical[col] = sorted(predictors, key=lambda x: x[1])
    
    # 5) Report MAR findings
    print("\nColumns with evidence of MAR (Missing At Random):")
    
    for col in missing_cols:
        num_predictors = mar_evidence_numeric.get(col, [])
        cat_predictors = mar_evidence_categorical.get(col, [])
        
        if num_predictors or cat_predictors:
            print(f"\n  {col} - Evidence of MAR:")
            
            if num_predictors:
                print("  Numeric variables that predict missingness:")
                for pred, p_val, diff_pct in num_predictors[:3]:
                    direction = "higher" if diff_pct > 0 else "lower"
                    print(f"    * {pred}: {abs(diff_pct):.1f}% {direction} when {col} is missing (p={p_val:.4f})")
            
            if cat_predictors:
                print("  Categorical variables that predict missingness:")
                for pred, p_val, effect in cat_predictors[:3]:
                    print(f"    * {pred}: Relationship with missingness (p={p_val:.4f}, effect size={effect:.2f})")
            
            all_predictors = [p[0] for p in (num_predictors + cat_predictors)[:3]]
            print(f"  Recommended approach: Use {', '.join(all_predictors)} as predictors in imputation")
        else:
            print(f"\n  {col} - No strong evidence of MAR detected")
            print("  Possibly MCAR or insufficient data to determine")
            print("  Recommended approach: Simple imputation (median, mode) or KNN")
    
    # 6) Check for potential MNAR indicators
    print("\nPotential indicators of MNAR (requires domain knowledge to confirm):")
    
    # Check for truncation patterns in distributions
    for col in numeric_cols:
        if df[col].isnull().any():
            # Get quantiles for non-missing data
            q = df[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).values
            # Check for unusual gaps or bunching near extremes
            if len(q) >= 7:
                ratio_upper = (q[6] - q[5]) / (q[5] - q[4])  # Ratio of 99-95 gap to 95-90 gap
                ratio_lower = (q[1] - q[0]) / (q[2] - q[1])  # Ratio of 25-10 gap to 50-25 gap
                
                if ratio_upper > 3 or ratio_upper < 0.33 or ratio_lower > 3 or ratio_lower < 0.33:
                    print(f"  * {col}: Unusual distribution near extremes, possible truncation pattern")
                    print(f"    This might indicate values are missing based on their magnitude (MNAR)")
    
    print("\nRecommendations for handling missing data:")
    print("1. For columns showing MAR patterns: Use regression imputation or MICE")
    print("2. For columns with no clear pattern: Use median/mode imputation")
    print("3. For columns with potential MNAR: Consider domain knowledge and sensitivity analysis")
    print("4. Consider flagging imputed values with indicator variables in downstream models")
    
    return {
        'missing_summary': missing_data,
        'mar_evidence_numeric': mar_evidence_numeric,
        'mar_evidence_categorical': mar_evidence_categorical
    }












def clean_dataset(df, columns_to_clean):
    """
    Clean the entire dataset iteratively, checking results after each cleaning step.
    """
    df_cleaned = df.copy()
    
    # First check - identify all problematic values
    print("Initial check for problematic values:")
    problematic_values = check_problematic_values(df_cleaned, columns_to_clean)
    
    # Define cleaning rules based on observed patterns
    cleaning_rules = {
        'remove_suffix': '_',
        'replace_comma': True,
        'remove_special_chars': ['$', '%', '__']
    }
    
    # Apply cleaning rules to each column
    for column in columns_to_clean:
        print(f"\nCleaning column: {column}")
        df_cleaned[column] = clean_numeric_values(df_cleaned, column, cleaning_rules)
    
    # Final check - verify if any problematic values remain
    print("\nChecking for remaining problematic values after cleaning:")
    remaining_issues = check_problematic_values(df_cleaned, columns_to_clean)
    
    return df_cleaned


def check_invalid_numeric_values(df, numeric_columns, valid_ranges):
    """Check for invalid numeric values"""
    for col in numeric_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        valid_min, valid_max = valid_ranges[col]
        
        invalid_mask = (df[col] < valid_min) | (df[col] > valid_max)
        if invalid_mask.any():
            print(f"\nInvalid values in {col}:")
            print(f"Range: {min_val} to {max_val}")
            print(f"Expected range: {valid_min} to {valid_max}")
            print(f"Number of invalid values: {invalid_mask.sum()}")
            print("\nExample invalid values:")
            print(df[col][invalid_mask].head())