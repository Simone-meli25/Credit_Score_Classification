import pandas as pd

def impute_missing_with_value_per_pattern(df, column_name, n_rows_per_pattern, method = 'unique'):
    """
    Imputes missing values in a column based on pattern analysis.
    
    Args:
        df: pandas dataframe
        column_name: The column to analyze
        n_rows_per_pattern: Number of rows that form a complete pattern
        method: Method to impute the missing values, either 'unique' or 'mean' or 'median' or 'mode'
            
    Returns:
        DataFrame with imputed values and status message
    """
    # Create a copy to avoid modifying the original dataframe
    df_imputed = df.copy()
    
    # Get column index for faster access
    col_idx = df_imputed.columns.get_loc(column_name)

    count_full_nan_blocks = 0
    
    # Impute using the unique value found in each block
    for i in range(0, len(df), n_rows_per_pattern):
        # Get the end index for this block (handle last block being shorter)
            end_idx = min(i + n_rows_per_pattern, len(df))
            
            # Get the current block
            block_values = df_imputed.iloc[i:end_idx][column_name]
            
            # Check if we have enough non-null values in this block
            non_null_values = block_values.dropna()
            
            if len(non_null_values) == 0:
                count_full_nan_blocks += 1
                continue
                
            unique_values = non_null_values.unique()

            if method == 'unique':
            
                if len(unique_values) != 1:
                    return df, (f"The column {column_name} cannot be imputed because the block of rows "
                            f"{i} - {end_idx-1} has {len(unique_values)} unique values, "
                            f"expected 1 with method = 'unique'")
                else:            
                    value_to_impute = unique_values[0]
            
            elif method == 'mean' and len(non_null_values) > 1:
                value_to_impute = non_null_values.mean()
            
            elif method == 'median' and len(non_null_values) > 1:
                value_to_impute = non_null_values.median()
            
            elif method == 'mode' and len(non_null_values) > 1:
                value_to_impute = non_null_values.mode()[0]
            
            # Impute missing values in this block
            for j in range(end_idx - i):
                if pd.isna(df_imputed.iloc[i+j][column_name]):
                    df_imputed.iloc[i+j, col_idx] = value_to_impute

    if count_full_nan_blocks > 0:
        print(f"""The missing values for column {column_name} have been imputed partially, using the method {method} per each block of {n_rows_per_pattern} rows. However there are {count_full_nan_blocks} blocks that are full of NaN values for column {column_name}, that have been left unchanged\n""")
    else:
        print(f"All the missing values for column {column_name} have been imputed, using the method {method} per each block of {n_rows_per_pattern} rows\n")     
    
    return df_imputed
    



def impute_missing_with_ordered_values_per_pattern(df, column_name, n_rows_per_pattern, ordered_values_to_impute=None):

    # Create a copy to avoid modifying the original dataframe
    df_imputed = df.copy()
    
    # Get column index for faster access
    col_idx = df_imputed.columns.get_loc(column_name)

    if isinstance(ordered_values_to_impute, pd.Series) and len(ordered_values_to_impute) == n_rows_per_pattern:
        # Impute using the provided pattern series
        for i in range(0, len(df), n_rows_per_pattern):
            # Get the end index for this block
            end_idx = min(i + n_rows_per_pattern, len(df))
            
            # Impute missing values in this block using the pattern
            for j in range(end_idx - i):
                if pd.isna(df_imputed.iloc[i+j][column_name]):
                        df_imputed.iloc[i+j, col_idx] = ordered_values_to_impute.iloc[j]
            
        print(f"All the missing values for column {column_name} have been imputed using the provided ordered values\n")     
        return df_imputed
    
    else:
        raise ValueError(f"The value for ordered_values_to_impute must be a pandas Series with length equal to n_rows_per_pattern ({n_rows_per_pattern})\n")
    



def impute_missing_with_incremental_value_per_pattern(df, column_name, n_rows_per_block, step_size=1/12):
    '''
    Impute missing values in a column based on incremental pattern analysis.
    Assumes the pattern has already been verified to be valid.
    '''
    df_imputed = df.copy()
    count_full_nan_blocks = 0

    # Start from index 0 and check every n_rows_per_block index
    for i in range(0, len(df), n_rows_per_block):
        # Get the block values as a Series
        block_values = pd.Series(df_imputed.loc[i:i+n_rows_per_block-1, column_name].values)

        if pd.isna(block_values).all():
            count_full_nan_blocks += 1
            continue

        non_nan_values = block_values[block_values.notna()]
        non_nan_indices = block_values[block_values.notna()].index

        # Find the first non-NaN value to use as reference
        first_valid_idx = non_nan_indices[0]
        first_valid_value = non_nan_values.iloc[0]

        # Impute values before the first valid value
        for j in range(first_valid_idx):
            if pd.isna(block_values.iloc[j]):
                df_imputed.loc[i+j, column_name] = first_valid_value - (step_size * (first_valid_idx - j))

        # Impute values between valid values
        for j in range(len(non_nan_values) - 1):
            current_idx = non_nan_indices[j]
            next_idx = non_nan_indices[j + 1]
            current_value = non_nan_values.iloc[j]
            
            for k in range(current_idx + 1, next_idx):
                if pd.isna(block_values.iloc[k]):
                    df_imputed.loc[i+k, column_name] = current_value + (step_size * (k - current_idx))
        
        # Impute values after the last valid value
        last_valid_idx = non_nan_indices[-1]
        last_valid_value = non_nan_values.iloc[-1]
        
        for j in range(last_valid_idx + 1, len(block_values)):
            if pd.isna(block_values.iloc[j]):
                df_imputed.loc[i+j, column_name] = last_valid_value + (step_size * (j - last_valid_idx))
    
    print(f"Imputation completed for column {column_name}:")
    print(f"- Full NaN blocks: {count_full_nan_blocks}")
    print(f"- Successfully imputed blocks: {len(df)//n_rows_per_block - count_full_nan_blocks}")
    
    return df_imputed