"""
YOU SHOULD INCLUDE IT IN THE CATEGORICAL FEATURES .PY FILE
"""




import pandas as pd
import numpy as np
import re


def convert_non_numeric_strings_to_nan(df, columns):
    """
    Converts strings that contain no numbers to NaN values.
    Preserves strings that contain at least one digit.
    
    Args:
        df: Input pandas DataFrame
        columns: List of column names to convert
        
    Returns:
        pandas DataFrame with strings containing no digits replaced by NaN
        
    Example:
        Input series: ["sbd", "se34d", "abc", "12", "xy45z"]
        Output series: [NaN, "se34d", NaN, "12", "xy45z"]
    """

    df_copy = df.copy()
    
    for col in columns:
        # Convert series to string type to ensure consistent processing
        df_copy[col] = df_copy[col].astype(str)
    
        # Create a boolean mask that is True when the string contains at least one digit
        has_digit_mask = df_copy[col].str.contains(r'\d')
        
        # Create a new series where strings without digits are replaced with NaN
        df_copy[col] = df_copy[col].where(has_digit_mask)
    
    return df_copy



def identify_problematic_characters(df, columns):
    """
    Identifies characters and special values that prevent numeric conversion in specified columns.
    
    Args:
        df: Input pandas DataFrame
        columns: List of column names to check
        
    Returns:
        Set containing problematic characters and special values
    """
    df_copy = df.copy()
    
    no_numeric_values = set()
    
    # First pass: identify all problematic characters
    for col in columns:
        if col not in df_copy.columns:
            print(f"Warning: Column '{col}' not found in DataFrame")
            continue
            
        # Convert to numeric with coercion (non-numeric becomes NaN) example: ['3', NaN, '_21'] --> [3, NaN, NaN]
        numeric_converted = pd.to_numeric(df_copy[col], errors='coerce')
    
        # True for values that aren't NaN in original but are NaN after conversion 
        # example: ['3', NaN, '_21'] --> [True, False, True] & [3, NaN, NaN] --> [False, True, True]
        mask = df_copy[col].notna() & numeric_converted.isna() 

        # Get values where mask is True
        problem_values = df_copy.loc[mask, col].astype(str)

        for val in problem_values:
            # Add whole string if it contains no digits
            if not any(c.isdigit() for c in val):
                no_numeric_values.add(val)
            else:
                # Find all characters that aren't digits, dot, or minus
                matches = re.findall(r'[^\d\.\-]', val)
                no_numeric_values.update(matches)

    # Second pass: print examples of values with problematic characters
    if no_numeric_values:
        # Generate regex pattern - only needed once after all characters are collected
        pattern = '|'.join(re.escape(c) for c in no_numeric_values)
        
        for col in columns:
                
            # Convert to string and find values containing problematic characters
            mask = df_copy[col].astype(str).str.contains(pattern, regex=True)
            problem_count = mask.sum()
            
            if problem_count > 0:
                print(f"\nColumn '{col}': {problem_count} values with problematic characters")
                print("-" * 50)
                
                # Get up to 3 examples
                examples = df_copy.loc[mask, col].astype(str).head(3)
                for i, ex in enumerate(examples):
                    print(f"Example {i+1}: '{ex}'")
            else:
                print(f"\nColumn '{col}': No problematic values found")
                
    return no_numeric_values


def remove_characters(df, columns, characters_to_remove='_'):
    """
    Remove specified characters from values in columns while preserving NaN values.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of column names to clean
        characters_to_remove (str or list, optional): Characters to remove. Can be a string 
                                           of multiple characters or a list of 
                                           individual characters. Defaults to '_'.
    
    Returns:
        pd.DataFrame: DataFrame with characters removed from specified columns
    """
    df_copy = df.copy()
    
    # Convert characters to list if it's a string
    if isinstance(characters_to_remove, str):
        char_list = list(characters_to_remove)
    else:
        char_list = characters_to_remove
    
    for column in columns:
        if column not in df_copy.columns:
            print(f"Warning: Column '{column}' not found in DataFrame")
            continue
            
        # Create a mask to identify NaN values
        nan_mask = df_copy[column].isna()
        
        # Process only non-NaN values
        non_nan_values = df_copy.loc[~nan_mask, column].astype(str)
        
        # Remove each character one by one
        for char in char_list:
            non_nan_values = non_nan_values.str.replace(char, '', regex=False)
        
        # Create a new series combining processed values and original NaN values
        new_series = pd.Series(index=df_copy.index, dtype=object)
        new_series.loc[~nan_mask] = non_nan_values
        new_series.loc[nan_mask] = np.nan
        
        # Update the column in the dataframe
        df_copy[column] = new_series
        
    return df_copy


def convert_to_numeric(df, columns):
    '''
    Convert specified columns to numeric type while preserving NaN values.
    
    '''
    df_copy = df.copy()

    for col in columns:
        if col in df_copy.columns:
            # Convert to numeric, coercing errors to NaN
            df_copy[col] = pd.to_numeric(df_copy[col], errors="raise")
            print(f"Converted '{col}' to numeric")
        else:
            print(f"Column '{col}' not found in DataFrame")

    return df_copy


