import pytest
import pandas as pd
import numpy as np
import os
import sys
import pathlib
import importlib


   

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




data = {
    'age': [25.0, 30.0, 35.0, np.nan, 40.0],
    'score': [100.0, 200.0, 300.0, 400.0, 500.0]
}

df = pd.DataFrame(data)

# Define constraints
constraints = {
    'age': {
        'min': 0,
        'max': 120,
        'integer': True
    },
    'score': {
        'min': 0,
        'max': 1000,
        'integer': True
    }
}


df_cleaned = set_constraints_for_numeric_features(df, 'age', constraints)

print(df_cleaned.head())
