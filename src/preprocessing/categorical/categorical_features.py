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
FUNCTIONS TO CLEAN CATEGORICAL FEATURES THAT HAVE PATTERNS WITH REGEX
'''

def clean_column_with_regex(df, column_name, pattern):
    """
    Validates values in a column against a regex pattern and replaces non-matching values with NaN.
    
    Args:
        df (pd.DataFrame): The dataframe to process
        column_name (str): The name of the column to validate
        pattern (str): The regex pattern to match against
        
    Returns:
        pd.DataFrame: A copy of the dataframe with non-matching values replaced with NaN
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Compile the regex pattern for better performance
    regex = re.compile(pattern)
    
    # Create a mask of values that don't match the pattern (excluding already null values)
    non_matching_mask = ~df_copy[column_name].apply(
        lambda x: bool(regex.match(str(x))) if pd.notna(x) else True
    )
    
    # Replace non-matching values with NaN
    df_copy.loc[non_matching_mask, column_name] = np.nan
    
    # Print summary of changes
    num_replaced = non_matching_mask.sum()
    print(f"Replaced {num_replaced} values in column '{column_name}' that didn't match pattern '{pattern}'\n")
    
    if num_replaced > 0:
        print("Sample of replaced values:")
        for idx, original_value in enumerate(df.loc[non_matching_mask, column_name].head(5)):
            print(f"  {original_value}")
        if num_replaced > 5:
            print(f"  ... and {num_replaced - 5} more\n")
    
    return df_copy


'''
FUNCTIONS TO ANALYZE PATTERNS FOR THE VALUES OF A CATEGORICAL FEATURE
'''


def analyze_unique_values_pattern(df, column_name, n_rows_per_block = 8, n_unique_values_per_block = 1):
    """
    Analyzes the dataset to determine if it follows a pattern consisting of 
    n_unique_values_per_block each n_rows_per_block
    
    Args:
        df: pandas dataframe
        column_name: The column to analyze

    Returns:
        Results of the analysis
    """
    
    count_full_nan_blocks = 0

    # Start from index 0 and check every n_rows_per_block index
    for i in range(0, len(df), n_rows_per_block):
        unique_values = df[column_name][i:i+n_rows_per_block].nunique()

        if unique_values == 0:
            count_full_nan_blocks += 1
            continue
        
        if unique_values == n_unique_values_per_block:
            continue
        elif unique_values != n_unique_values_per_block:
            return f"The dataset for column {column_name} is not organized in blocks of {n_rows_per_block} records, since the block of rows {i} - {i+n_rows_per_block -1} has {unique_values} unique values, with {count_full_nan_blocks} blocks that are full of NaN values\n"

    return f"The dataset for column {column_name} is organized in blocks of {n_rows_per_block} records, since each block of {n_rows_per_block} rows has {unique_values} unique values, with {count_full_nan_blocks} blocks that are full of NaN values\n"



def analyze_ordered_values_pattern(df, column_name, n_rows_per_block, ordered_sequence):

    '''
    This function checks if the dataset for column_name is organized in blocks of n_rows_per_block, 
    where each block of n_rows_per_block rows is ordered according to the ordered_sequence.
    '''

    count_full_nan_blocks = 0

    # Start from index 0 and check every n_rows_per_block index
    for i in range(0, len(df), n_rows_per_block):
        block_values = df[column_name][i:i+n_rows_per_block].values

        if pd.isna(block_values).all():
            count_full_nan_blocks += 1
            continue

        for j in range(n_rows_per_block):
            if pd.isna(block_values[j]):
                continue
            if block_values[j] != ordered_sequence[j]:
                return f"The dataset for column {column_name} is not organized in blocks of {n_rows_per_block} ordered records \n"

    return f"The dataset for column {column_name} is organized in blocks of {n_rows_per_block} ordered records, since each block of {n_rows_per_block} rows have a perfect correspondence, with {count_full_nan_blocks} blocks that are full of NaN values\n"






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
    Visualizes the top most frequent categories for the column, including missing values as a category.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data
        column (str): The column name to analyze
        top_n (int): Number of top categories to display (default: 35)
        
    Returns:
        pd.Series: The top categories with their frequencies
    """
    # Count occurrences of each category for the column INCLUDING missing values
    categories_counts = df[column].value_counts(dropna=False)
    
    # Calculate percentages
    total_records = len(df)
    categories_percentages = categories_counts / total_records * 100
    
    # Create a DataFrame with counts and percentages (instead of dictionary since we have thousands of categories)
    categories_stats = pd.DataFrame({
        'Count': categories_counts,
        'Percentage': categories_percentages
    }).reset_index()
    categories_stats.columns = [column, 'Count', 'Percentage']
    
    # Replace NaN with "Missing" in the category names
    categories_stats[column] = categories_stats[column].fillna('Missing')
    
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
    
    '''
    # No need for separate missing values info as it's now included in the categories
    # but we can highlight it explicitly if it's in the top categories
    if 'Missing' in top_categories[column].values:
        missing_row = top_categories[top_categories[column] == 'Missing']
        if not missing_row.empty:
            missing_count = missing_row['Count'].values[0]
            missing_pct = missing_row['Percentage'].values[0]
            print(f"\nNote: Missing values ({missing_count:,d} records, {missing_pct:.2f}%) are included as a category.")
    '''


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

        # Create a binary column for the unique value
        df_copy[col_name] = (
            df_copy[column].str.contains(re.escape(unique_value), case=True, regex=True).astype(int)
        )

    # Add count column
    df_copy[f'{feature_name}_Count'] = df_copy[column].str.count('|'.join(map(re.escape, unique_values)))
      
    return df_copy

'''
FUNCTIONS TO HANDLE "CREDIT HISTORY AGE" FEATURE
'''

def parse_string_time_period(value, pattern):
    """
    Parse a single time period string into a numeric value.
    
    Args:
        value: The string value to parse
        pattern (str): The regex pattern to match against
    Returns:
        float: Time period in decimal years, or np.nan if invalid
    """
    if pd.isna(value):
        return np.nan
    if not isinstance(value, str):
        return np.nan
        
    match = re.search(pattern, value, re.IGNORECASE)
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years + months/12
    return np.nan




    


