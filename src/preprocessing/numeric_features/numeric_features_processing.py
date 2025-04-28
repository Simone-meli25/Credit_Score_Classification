import numpy as np


def set_constraints_for_numeric_features(df, numeric_columns, constraints):
    """
    Handle inconsistent numeric values in a numeric feature by:
    1. Min values when inappropriate
    2. Max values when inappropriate
    3. Decimal values when integers expected

    Args:
        df (pd.DataFrame): Input dataframe
        numeric_columns (list): List of numeric columns to process
        constraints (dict): Dictionary of constraints for each feature
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """

    df_copy = df.copy()

    # Validate constraints dictionary first - fail fast
    REQUIRED_KEYS = ['min', 'max', 'integer']

    missing_keys = []

    for column in numeric_columns:
        # Check if feature has constraints defined
        if column not in constraints:
            raise ValueError(f"No constraints defined for column '{column}'")
            
        # Check if all required keys exist for this column
        for key in REQUIRED_KEYS:
            if key not in constraints[column]:
                missing_keys.append(key)
        
        # Raise error if any constraints are missing
        if missing_keys:
            error_msg = f"Missing constraints for column '{column}': {', '.join(missing_keys)}"
            raise ValueError(error_msg)
    
        # If we get here, all constraints are properly defined

        print(f"\nCleaning {column}...")
        
        # 1. Handle below-minimum values
        if constraints[column]['min'] is not None:
            below_min_mask = df_copy[column] < constraints[column]['min']
            if below_min_mask.any():
                print(f"  Found {below_min_mask.sum()} below-minimum constraint. Replacing with NaN")
                df_copy.loc[below_min_mask, column] = np.nan

        # 2. Handle above-maximum values
        if constraints[column]['max'] is not None:
            above_max_mask = df_copy[column] > constraints[column]['max']
            if above_max_mask.any():
                print(f"  Found {above_max_mask.sum()} above-maximum constraint. Replacing with NaN")
                df_copy.loc[above_max_mask, column] = np.nan
        
        # 2. Handle decimal values for integer features
        if constraints[column]['integer']:
            decimal_mask = df_copy[column].notna() & (df_copy[column] != df_copy[column].round())
            if decimal_mask.any():
                print(f"  Found {decimal_mask.sum()} decimal values in integer feature. Rounding to nearest integer")
                df_copy.loc[decimal_mask, column] = df_copy.loc[decimal_mask, column].round()
                # Convert whole column to integer type
                df_copy[column] = df_copy[column].astype('Int64')  # Handles NaN values
    
    return df_copy