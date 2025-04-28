
'''
MISSING VALUES FOR CATEGORICAL FEATURES
'''
import pandas as pd

def visualize_proportion_of_unique_categories(df, columns):
    """
    For each column in `columns`, show the proportion of unique categories (including missing values).

    """
    for col in columns:
        if col not in df.columns:
            print(f"Column '{col}' not found\n")
            continue

        # get the non-null uniques
        vals = df[col].unique().tolist()
        print(f"\nColumn '{col}' has {len(vals)} unique categories:\n")
       
        # Calculate value counts including missing values
        value_counts = df[col].value_counts(dropna=False)
        total_count = len(df)
        
        # Calculate proportions
        proportions = value_counts / total_count
        
        if len(proportions) > 10:
            print(f"Showing top 10 most frequent categories (proportion) for column '{col}':\n")
            for val, prop in proportions.head(10).items():
                val_display = "Missing" if pd.isna(val) else val
                print(f"{val_display}: {(prop*100):.2f}%")
        else:
            print(f"All categories (proportion) for column '{col}':\n")
            for val, prop in proportions.items():
                val_display = "Missing" if pd.isna(val) else val
                print(f"{val_display}: {(prop*100):.2f}%")

        print("\n"+"-"*100+"\n")




def impute_missing_values_for_categorical_features(df, categorical_columns, methods):
    """
    Impute missing values in categorical features using the specified method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorical_columns (list): List of categorical columns to process
        methods (dict): Dictionary of methods for each column
        
    Returns:
        pd.DataFrame: Dataframe with imputed values
    """
    df_copy = df.copy()

    ALLOWED_METHODS = ['knn', 'unknown_category']

    for column in categorical_columns:
        # Check if feature has constraints defined
        if column not in methods:
            raise ValueError(f"No method defined for column '{column}'")
        
        if methods[column] not in ALLOWED_METHODS:
            raise ValueError(f"Invalid method for column '{column}': {methods[column]}")
    
        if not df_copy[column].isna().any():
            print(f"No missing values to impute for {column}")
            return df_copy
        
        column_missing_count = df_copy[column].isna().sum()
    
        print(f"Imputing {column_missing_count} missing values for {column} using '{methods[column]}'")
    
        if methods[column] == 'knn':
            # Use KNN imputation
            continue

            
        elif methods[column] == 'unknown_category':
            # Create a new category for missing values
            df_copy[column] = df_copy[column].fillna('Unknown')
            print(f"    Using 'Unknown' as a new category")
    
    return df_copy










'''
MISSING VALUES FOR NUMERICAL FEATURES
'''