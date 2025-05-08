import numpy as np
import pandas as pd

def set_constraints_for_numeric_features(df, numeric_column, constraints):
    """
    Handle inconsistent numeric values in a numeric feature by:
    1. Min values when inappropriate
    2. Max values when inappropriate
    3. Decimal values when integers expected

    Args:
        df (pd.DataFrame): Input dataframe
        numeric_column (str): Name of the numeric column to process
        constraints (dict): Dictionary of constraints for the feature
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """

    df_copy = df.copy()

    # Validate constraints dictionary first - fail fast
    REQUIRED_KEYS = ['min', 'max', 'integer']

    missing_keys = []

    # Check if feature has constraints defined
    if numeric_column not in constraints:
        raise ValueError(f"No constraints defined for column '{numeric_column}'")
        
    # Check if all required keys exist for this column
    for key in REQUIRED_KEYS:
        if key not in constraints[numeric_column]:
            missing_keys.append(key)
    
    # Raise error if any constraints are missing
    if missing_keys:
        error_msg = f"Missing constraints for column '{numeric_column}': {', '.join(missing_keys)}"
        raise ValueError(error_msg)

    # If we get here, all constraints are properly defined

    print(f"\nCleaning {numeric_column}...")
    
    # 1. Handle below-minimum values
    if constraints[numeric_column]['min'] is not None:
        below_min_mask = df_copy[numeric_column] < constraints[numeric_column]['min']
        if below_min_mask.any():
            print(f"  Found {below_min_mask.sum()} below-minimum constraint. Replacing with NaN")
            df_copy.loc[below_min_mask, numeric_column] = np.nan

    # 2. Handle above-maximum values
    if constraints[numeric_column]['max'] is not None:
        above_max_mask = df_copy[numeric_column] > constraints[numeric_column]['max']
        if above_max_mask.any():
            print(f"  Found {above_max_mask.sum()} above-maximum constraint. Replacing with NaN")
            df_copy.loc[above_max_mask, numeric_column] = np.nan
    
    # 2. Handle decimal values for integer features
    if constraints[numeric_column]['integer']:
        df_copy[numeric_column] = df_copy[numeric_column].astype('Int64')
    
    return df_copy



def analyze_incremental_num_values_pattern(df, column_name, n_rows_per_block, step_size=1/12, tolerance=0.001):

    '''
    This function checks if the dataset for column_name is organized in blocks of n_rows_per_block, 
    where each block of n_rows_per_block rows is ordered according to the ordered_sequence.
    '''

    count_full_nan_blocks = 0

    # Start from index 0 and check every n_rows_per_block index
    for i in range(0, len(df), n_rows_per_block):
        block_values = pd.Series(df[column_name][i:i+n_rows_per_block].values)

        if pd.isna(block_values).all():
            count_full_nan_blocks += 1
            continue

        non_nan_values = block_values[block_values.notna()]
        non_nan_indices = block_values[block_values.notna()].index

        # Check if we have at least 2 non-NaN values
        if len(non_nan_values) < 2:
            return f"One block of {n_rows_per_block} rows does not have at least 2 non-NaN values. It is not possible to check if the dataset is organized in blocks of {n_rows_per_block} ordered records with incremental step of {step_size} \n"

        for j in range(len(non_nan_values) - 1):
            current_value = non_nan_values.iloc[j]
            next_value = non_nan_values.iloc[j + 1]
            current_idx = non_nan_indices[j]
            next_idx = non_nan_indices[j + 1]
            
            # Calculate expected step based on number of positions between values
            expected_step = step_size * (next_idx - current_idx)
            
            # Check if the actual difference is within tolerance of expected step
            actual_diff = next_value - current_value
            if not (expected_step - tolerance <= actual_diff <= expected_step + tolerance):
                return f"The dataset for column {column_name} is not organized in blocks of {n_rows_per_block} ordered records with incremental step of {step_size} \n"
    
    return f"The dataset for column {column_name} is organized in blocks of {n_rows_per_block} ordered records with incremental step of {step_size} and {count_full_nan_blocks} blocks that are full of NaN values\n"
