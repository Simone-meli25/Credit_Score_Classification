import re
import pandas as pd
import numpy as np

'''
FUNCTIONS TO VISUALIZE UNIQUE CATEGORIES
'''

def visualize_unique_categories(df, columns):
    """
    For each column in `columns`, print and return its non-missing unique values.

    """
    for col in columns:
        if col not in df.columns:
            print(f"Column '{col}' not found")
            continue

        # get the non-null uniques
        vals = df[col].dropna().unique().tolist()
        print(f"\nColumn '{col}' ({len(vals)} uniques):")
        if len(vals) > 35:
            print("the length of the list is too long to be printed entirely. Sample:")
            print(vals[:10])
        else:
            print(vals)
    
    return 


'''
FUNCTIONS TO HANDLE "STREET" FEATURE
'''

def add_space_before_word(df, feature_column, separator_word):
    """
    Ensure that the feature_name is separated by a space from the separator_word.
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Only process non-null values
    mask = df_copy[feature_column].notna()
    
    # Use regex to insert a space before 'Street' if missing
    df_copy.loc[mask, feature_column] = df_copy.loc[mask, feature_column].apply(
        lambda x: re.sub(r'(\w+)(?<!\s)' + separator_word, r'\1 ' + separator_word, x) if isinstance(x, str) else x
    )
    
    return df_copy


'''
FUNCTIONS TO HANDLE "TYPE OF LOAN" FEATURE
'''

def visualize_top_n_categories(df, column, top_n=35):
    """
    Visualizes the top most frequent categories for the column.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data
        column (str): The column name to analyze
        top_n (int): Number of top categories to display (default: 35)
        
    Returns:
        pd.Series: The top categories with their frequencies
    """
    # Count occurrences of each category for the column
    categories_counts = df[column].value_counts()
    
    # Calculate percentages
    total_records = len(df)
    categories_percentages = categories_counts / total_records * 100
    
    # Create a DataFrame with counts and percentages
    categories_stats = pd.DataFrame({
        'Count': categories_counts,
        'Percentage': categories_percentages
    }).reset_index()
    categories_stats.columns = [column, 'Count', 'Percentage']
    
    # Get the top N categories
    top_categories = categories_stats.head(top_n)
    
    # Display results
    print(f"\nTop {top_n} Categories (out of {categories_counts.shape[0]} unique values):")
    print(f"Total records analyzed: {total_records}")
    print("\nDetailed breakdown:")
    
    # Format the output for better readability
    for i, (_, row) in enumerate(top_categories.iterrows(), 1):
        category = row[column]
        count = row['Count']
        percentage = row['Percentage']
        print(f"{i:2d}. {category[:70]:<70} {count:6,d} records ({percentage:.2f}%)")
    
    # Show cumulative percentage covered by top N categories
    cumulative_percentage = top_categories['Percentage'].sum()
    print(f"\nThe top {top_n} categories cover {cumulative_percentage:.2f}% of all records")
    
    # Print info about missing values
    missing_count = df[column].isna().sum()
    missing_pct = missing_count / total_records * 100
    print(f"Missing values: {missing_count:,d} ({missing_pct:.2f}%)")
    


def get_unique_values_and_counts(df, column, split_pattern=r'\s*(?:,|and)\s*'):
    """
    Extracts individual values from a column containing comma/and-separated lists,
    then counts the frequency of each unique value.
    
    Args:
        df: DataFrame containing the data
        column: Name of the column containing separated values to analyze
        split_pattern: Regex pattern used to split values (default: split on commas or "and")
        
    Returns:
        Dictionary of unique values and their counts, sorted by frequency (descending)
    """
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Initialize counter dictionary
    unique_values_and_counts = {}
    
    # Count occurrences of each individual value
    for value in df_copy[column].dropna():
        # Split on pattern (commas or "and" with surrounding spaces)
        parts = re.split(split_pattern, value)
        # Clean and deduplicate parts
        unique_values = {p.strip() for p in parts if p.strip()}
        # Count each unique value
        for unique_value in unique_values:
            unique_values_and_counts[unique_value] = unique_values_and_counts.get(unique_value, 0) + 1
    
    # Sort by frequency (descending)
    sorted_unique_values_and_counts = dict(sorted(
        unique_values_and_counts.items(),
        key=lambda item: item[1],
        reverse=True
    ))
    
    return sorted_unique_values_and_counts



def transform_to_binary_features(df, column, unique_values, feature_name = 'Loan', suffix = 'Has_'):
    """
    Transforms the column into multiple binary features
    based on the presence of basic unique values.
    """
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    print(f"\nCreating binary features for {len(unique_values)} unique values")
    
    # Create binary columns for each basic loan type
    for unique_value in unique_values:

        # Create a clean column name
        col_name = suffix + re.sub(r'[ ,&-]+', '_', unique_value)

        # Handle missing values by filling NaN with empty strings before checking (this part could be not necessary if we handle missing values before)
        df_copy[col_name] = (
            df_copy[column].fillna('').str.contains(re.escape(unique_value), case=True, regex=True).astype(int)
        )
        
    return df_copy

'''
FUNCTIONS TO HANDLE "CREDIT HISTORY AGE" FEATURE
'''

def parse_string_time_period(string_time_period, pattern = r"(\d+)\s*(?:Years?|Yrs?)\s*(?:and)?\s*(\d+)\s*(?:Months?|Mon)"):
    """
    Parse a column containing time period strings into a numeric values.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data
        column (str): The column name to analyze
        
    Returns:
        float: Time period in decimal years (e.g., 22.08 for '22 Years and 1 Months')
    """
    if pd.isna(string_time_period) or not isinstance(string_time_period, str):
        return np.nan
        
    # More flexible regex pattern to handle variations
    match = re.search(pattern, string_time_period, re.IGNORECASE)
    
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years + months/12
    else:
        # Log problematic values for inspection
        print(f"Warning: Could not parse time period for column: '{string_time_period}'")
        return np.nan


''' I should add also for type of loans'''


    


