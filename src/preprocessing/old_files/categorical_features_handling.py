import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns



def list_unique_categories(df, columns):
    """
    For each column in `columns`, print and return its non-missing unique values.
    Returns a dict mapping column → list of uniques.
    """
    uniques = {}
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
        #uniques[col] = vals
    return uniques


def fix_street_spacing(df):
    """
    Ensure that 'Street' is separated by a space in the Street column.
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Only process non-null values
    mask = df_copy['Street'].notna()
    
    # Use regex to insert a space before 'Street' if missing
    df_copy.loc[mask, 'Street'] = df_copy.loc[mask, 'Street'].apply(
        lambda x: re.sub(r'(\w+)(?<!\s)Street', r'\1 Street', x) if isinstance(x, str) else x
    )
    
    return df_copy



def parse_credit_history_age(age_str):
    """
    Parse credit history age from format like '22 Years and 1 Months' to decimal years.
    
    Args:
        age_str (str): Credit history age string
        
    Returns:
        float: Age in decimal years (e.g., 22.08 for '22 Years and 1 Months')
    """
    if pd.isna(age_str) or not isinstance(age_str, str):
        return np.nan
        
    # More flexible regex pattern to handle variations
    pattern = r"(\d+)\s*(?:Years?|Yrs?)\s*(?:and)?\s*(\d+)\s*(?:Months?|Mon)"
    match = re.search(pattern, age_str, re.IGNORECASE)
    
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years + months/12
    else:
        # Log problematic values for inspection
        print(f"Warning: Could not parse credit history age: '{age_str}'")
        return np.nan
    
    



def check_category_percentages(df, column):
    """
    Calculate the percentage of each category in a column, including NaN values.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to analyze
        
    Returns:
        pd.Series: Percentages for each category, including NaN
    """
    # Get total count
    total_count = len(df)
    
    # Create a series that includes NaN values
    # First get counts of all non-null values
    value_counts = df[column].value_counts(dropna=False)
    
    # Calculate percentages
    percentages = value_counts / total_count * 100
    
    # Print the results
    print(f"\nCategory distribution for '{column}':")
    print(f"Total records: {total_count}")
    
    # Format with value counts and percentages
    for category, count in value_counts.items():
        category_name = 'NaN' if pd.isna(category) else category
        percentage = percentages[category]
        print(f"{category_name}: {count} records ({percentage:.2f}%)")
    
    return percentages



def analyze_loan_types(df, column='Type_of_Loan', top_n=35):
    """
    Analyzes the top most frequent categories for the Type_of_Loan feature.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data
        column (str): The column name to analyze (default: 'Type_of_Loan')
        top_n (int): Number of top categories to display (default: 35)
        
    Returns:
        pd.Series: The top categories with their frequencies
    """
    # Count occurrences of each loan type
    loan_counts = df[column].value_counts()
    
    # Calculate percentages
    total_records = len(df)
    loan_percentages = loan_counts / total_records * 100
    
    # Create a DataFrame with counts and percentages
    loan_stats = pd.DataFrame({
        'Count': loan_counts,
        'Percentage': loan_percentages
    }).reset_index()
    loan_stats.columns = ['Loan Type', 'Count', 'Percentage']
    
    # Get the top N categories
    top_loans = loan_stats.head(top_n)
    
    # Display results
    print(f"\nTop {top_n} Loan Types (out of {loan_counts.shape[0]} unique values):")
    print(f"Total records analyzed: {total_records}")
    print("\nDetailed breakdown:")
    
    # Format the output for better readability
    for i, (_, row) in enumerate(top_loans.iterrows(), 1):
        loan_type = row['Loan Type']
        count = row['Count']
        percentage = row['Percentage']
        print(f"{i:2d}. {loan_type[:70]:<70} {count:6,d} records ({percentage:.2f}%)")
    
    # Show cumulative percentage covered by top N categories
    cumulative_percentage = top_loans['Percentage'].sum()
    print(f"\nThe top {top_n} loan types cover {cumulative_percentage:.2f}% of all records")
    
    # Print info about missing values
    missing_count = df[column].isna().sum()
    missing_pct = missing_count / total_records * 100
    print(f"Missing values: {missing_count:,d} ({missing_pct:.2f}%)")
    
    return top_loans



def transform_loan_types(df, column='Type_of_Loan', min_frequency=10):
    """
    Transforms the Type_of_Loan column into multiple binary features
    based on the presence of basic loan types.
    
    Args:
        df: DataFrame containing the loan data
        column: Name of the column containing loan types
        min_frequency: Minimum frequency for a loan type to be considered 'basic'
    """
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Discover all unique loan types by splitting the combinations
    all_loan_types = {}
    
    # Count occurrences of each loan type
    for value in df[column].dropna():
        parts = re.split(r'\s*(?:,|and)\s*', value) # split on commas or the word “and” (with surrounding spaces)
        loans = {p.strip() for p in parts if p.strip()} # clean & dedupe
        for loan in loans:
            all_loan_types[loan] = all_loan_types.get(loan, 0) + 1


        
    
    # Filter to include only loan types with minimum frequency
    frequent_loan_types = {loan: count for loan, count in all_loan_types.items() if count >= min_frequency}
    
    # Sort by frequency (descending)
    frequent_loan_types = dict(sorted(frequent_loan_types.items(), 
                                      key=lambda item: item[1], 
                                      reverse=True))
    
    # Print the discovered loan types and their frequencies
    print(f"Discovered {len(frequent_loan_types)} loan types with at least {min_frequency} occurrences:")
    for loan_type, count in list(frequent_loan_types.items())[:20]:  # Show top 20
        print(f"- {loan_type}: {count:,} occurrences")
    if len(frequent_loan_types) > 20:
        print(f"... and {len(frequent_loan_types) - 20} more loan types")
    
    # Use the frequent loan types as our basic loan types
    basic_loan_types = list(frequent_loan_types.keys())
    
    print(f"\nCreated binary features for {len(basic_loan_types)} loan types")
    
    # Create binary columns for each basic loan type
    for loan_type in basic_loan_types:

        # Create a clean column name
        col_name = 'Has_' + re.sub(r'[ ,&-]+', '_', loan_type)

        # Handle missing values by filling NaN with empty strings before checking
        df_copy[col_name] = (
            df_copy[column].fillna('').str.contains(re.escape(loan_type), case=True, regex=True).astype(int)
        )
        
    
    # Create a column for truly 'other' loan types not in our expanded basic list
    df_copy['Has_Other_Loan_Type'] = 0
    other_count = 0
    
    # Check each row for loan types not in our expanded basic list
    for idx, value in df_copy[column].dropna().items():
        parts = re.split(r'\s*(?:,|and)\s*', value)
        loans = {p.strip() for p in parts if p.strip()}
        if not loans.issubset(basic_loan_types):
            df_copy.at[idx, 'Has_Other_Loan_Type'] = 1
            other_count += 1
    if other_count > 0:
        print(f"\n{other_count:,} rows have at least one 'other' loan type")
    else:
        df_copy.drop(columns=['Has_Other_Loan_Type'], inplace=True)
    
    
    # Create a count of loan types per customer
    df_copy['Loan_Type_Count'] = df_copy[column].fillna('').apply(
        lambda x: 0 if not x else len({p.strip() for p in re.split(r'\s*(?:,|and)\s*', x) if p.strip()})
        )

    
    return df_copy





def analyze_loan_type_patterns(df, column='Type_of_Loan'):
    """
    Analyzes patterns in loan type combinations
    """
    # Count frequency of each basic loan type
    basic_loan_types = [
        'Not Specified', 'Credit-Builder Loan', 'Debt Consolidation Loan', 
        'Personal Loan', 'Student Loan', 'Payday Loan', 'Mortgage Loan', 
        'Auto Loan', 'Home Equity Loan'
    ]
    
    # Count occurrences of each basic type (including when part of combinations)
    loan_type_counts = {loan_type: 0 for loan_type in basic_loan_types}
    
    for value in df[column].dropna():
        for loan_type in basic_loan_types:
            if loan_type in value:
                loan_type_counts[loan_type] += 1
    
    # Sort by frequency
    loan_type_counts = {k: v for k, v in sorted(
        loan_type_counts.items(), key=lambda item: item[1], reverse=True)}
    
    # Print results
    print("Frequency of each basic loan type (including in combinations):")
    for loan_type, count in loan_type_counts.items():
        print(f"{loan_type}: {count:,} ({count/len(df)*100:.2f}%)")
    
    # Analyze combination patterns
    combo_sizes = df[column].fillna('').apply(
        lambda x: 0 if x == '' else 1 + x.count(', and ')).value_counts()
    
    print("\nDistribution of number of loan types per customer:")
    for size, count in sorted(combo_sizes.items()):
        print(f"{size} loan type(s): {count:,} records ({count/len(df)*100:.2f}%)")





def check_problematic_categorical_values(df, columns, valid_categories=None):
    """
    Check and display problematic categorical values in specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of categorical column names to check
        valid_categories (dict, optional): Dictionary with column names as keys and 
                                          lists of valid categories as values
        
    Returns:
        dict: Dictionary with problematic values for each column
    """
    problematic_values = {}
    
    for col in columns:
        if col not in df.columns:
            print(f"Column '{col}' not found in dataframe")
            continue
            
        # Check for missing values
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        print(f"\nColumn '{col}':")
        print(f"Missing values: {missing_count} ({missing_pct:.2f}%)")
        
        # Get value counts for non-missing values
        value_counts = df[col].value_counts()
        unique_count = len(value_counts)
        print(f"Unique values: {unique_count}")
        
        # Check if there are too many unique values (potential issue for categorical)
        if unique_count > 100:
            print(f"WARNING: High cardinality - {unique_count} unique values")
            print("This may not be a true categorical variable or might need grouping")
            print("Top 10 most common values:")
            print(value_counts.head(10))
            problematic_values[col] = {"high_cardinality": True, "unique_count": unique_count}
            continue
        
        # Check for rare categories (appearing in less than 1% of non-missing data)
        non_missing_count = len(df) - missing_count
        rare_categories = value_counts[value_counts / non_missing_count < 0.01]
        if not rare_categories.empty:
            print(f"Rare categories (< 1% of data) - consider grouping:")
            print(rare_categories)
            if col not in problematic_values:
                problematic_values[col] = {}
            problematic_values[col]["rare_categories"] = rare_categories.index.tolist()
        
        # Check for invalid categories if valid_categories is provided
        if valid_categories and col in valid_categories:
            valid_set = set(valid_categories[col])
            actual_set = set(df[col].dropna().unique())
            invalid_categories = actual_set - valid_set
            
            if invalid_categories:
                print(f"Invalid categories detected:")
                invalid_counts = df[col][df[col].isin(invalid_categories)].value_counts()
                print(invalid_counts)
                if col not in problematic_values:
                    problematic_values[col] = {}
                problematic_values[col]["invalid_categories"] = invalid_counts.to_dict()
        
        # Check for potential case inconsistencies (for string categories)
        if df[col].dtype == 'object':
            # Get lowercase version of all values
            lowercase_values = df[col].dropna().str.lower()
            # Get unique lowercase values
            unique_lowercase = lowercase_values.unique()
            
            # Check if we have fewer unique lowercase values than original unique values
            if len(unique_lowercase) < unique_count:
                print("Potential case inconsistencies detected:")
                case_issues = {}
                for lower_val in unique_lowercase:
                    matches = [val for val in df[col].unique() if val is not None and val.lower() == lower_val]
                    if len(matches) > 1:
                        case_issues[lower_val] = matches
                
                for lower_val, variants in case_issues.items():
                    print(f"  '{lower_val}' appears as: {variants}")
                
                if col not in problematic_values:
                    problematic_values[col] = {}
                problematic_values[col]["case_inconsistencies"] = case_issues
                
        # Check for leading/trailing whitespaces
        if df[col].dtype == 'object':
            has_spaces = (df[col].notna() & 
                          ((df[col].str.len() != df[col].str.strip().str.len())))
            if has_spaces.any():
                print("Values with leading/trailing whitespaces detected:")
                space_issues = df.loc[has_spaces, col].value_counts()
                print(space_issues)
                if col not in problematic_values:
                    problematic_values[col] = {}
                problematic_values[col]["whitespace_issues"] = space_issues.to_dict()
    
    return problematic_values


def standardize_categorical_values(df, column, standardization_rules=None):
    """
    Standardize categorical values based on specified rules.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to clean
        standardization_rules (dict, optional): Dictionary of standardization rules
            Example: {
                'case': 'lower',  # or 'upper', 'title'
                'strip_whitespace': True,
                'replace_values': {'M': 'Male', 'F': 'Female'},
                'group_rare': 0.01,  # Group categories below this frequency threshold
                'group_name': 'Other'  # Name for the grouped category
            }
    
    Returns:
        pd.Series: Cleaned categorical column
    """
    df_copy = df.copy()
    series = df_copy[column].copy()
    
    if standardization_rules is None:
        print(f"No standardization rules provided for {column}")
        return series

    # Apply standardization rules
    # 1. Case standardization
    if 'case' in standardization_rules and series.dtype == 'object':
        if standardization_rules['case'] == 'lower':
            series = series.str.lower()
            print(f"  Converted {column} to lowercase")
        elif standardization_rules['case'] == 'upper':
            series = series.str.upper()
            print(f"  Converted {column} to uppercase")
        elif standardization_rules['case'] == 'title':
            series = series.str.title()
            print(f"  Converted {column} to title case")
    
    # 2. Strip whitespace
    if standardization_rules.get('strip_whitespace', False) and series.dtype == 'object':
        series = series.str.strip()
        print(f"  Stripped whitespace from {column}")
    
    # 3. Replace values
    if 'replace_values' in standardization_rules:
        replacements = standardization_rules['replace_values']
        series = series.replace(replacements)
        print(f"  Replaced values in {column}: {replacements}")
    
    # 4. Group rare categories
    if 'group_rare' in standardization_rules:
        threshold = standardization_rules['group_rare']
        group_name = standardization_rules.get('group_name', 'Other')
        
        # Calculate frequencies
        value_counts = series.value_counts(normalize=True)
        rare_categories = value_counts[value_counts < threshold].index.tolist()
        
        if rare_categories:
            series = series.apply(lambda x: group_name if x in rare_categories else x)
            print(f"  Grouped {len(rare_categories)} rare categories as '{group_name}'")
    
    # 5. Handle missing values
    if 'missing_strategy' in standardization_rules:
        strategy = standardization_rules['missing_strategy']
        
        if strategy == 'mode':
            mode_value = series.mode()[0]
            series = series.fillna(mode_value)
            print(f"  Filled missing values with mode: '{mode_value}'")
        
        elif strategy == 'custom':
            custom_value = standardization_rules.get('missing_value', 'Unknown')
            series = series.fillna(custom_value)
            print(f"  Filled missing values with custom value: '{custom_value}'")
    
    return series


def handle_categorical_features(df, categorical_feature, constraints):
    """
    Handle categorical feature issues by:
    1. Standardizing case and format
    2. Handling invalid or rare categories
    3. Imputing missing values appropriately
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorical_feature (str): Name of the categorical column to process
        constraints (dict): Dictionary of constraints for this feature
            Example: {
                'valid_categories': ['A', 'B', 'C'],
                'case': 'title',
                'strip_whitespace': True,
                'replace_values': {'M': 'Male', 'F': 'Female'},
                'handle_rare': True,
                'rare_threshold': 0.01,
                'rare_group_name': 'Other',
                'missing_strategy': 'mode'  # or 'custom'
                'missing_value': 'Unknown'  # used if missing_strategy is 'custom'
            }
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Validate constraints dictionary first - fail fast
    required_keys = ['valid_categories', 'missing_strategy']
    
    # Check if feature has constraints defined
    if categorical_feature not in constraints:
        raise ValueError(f"No constraints defined for column '{categorical_feature}'")
            
    # Check if all required keys exist for this column
    missing_keys = []
    for key in required_keys:
        if key not in constraints[categorical_feature]:
            missing_keys.append(key)
    
    # Raise error if any required constraints are missing
    if missing_keys:
        error_msg = f"Missing constraints for column '{categorical_feature}': {', '.join(missing_keys)}"
        raise ValueError(error_msg)
    
    # If we get here, all constraints are properly defined
    df_clean = df.copy()
    
    print(f"\nCleaning categorical feature: {categorical_feature}")
    
    # 1. Create standardization rules from constraints
    rules = {
        'case': constraints[categorical_feature].get('case', None),
        'strip_whitespace': constraints[categorical_feature].get('strip_whitespace', True),
        'replace_values': constraints[categorical_feature].get('replace_values', {}),
        'missing_strategy': constraints[categorical_feature]['missing_strategy']
    }
    
    # Add custom missing value if strategy is 'custom'
    if rules['missing_strategy'] == 'custom':
        rules['missing_value'] = constraints[categorical_feature].get('missing_value', 'Unknown')
    
    # Add grouping for rare categories if needed
    if constraints[categorical_feature].get('handle_rare', False):
        rules['group_rare'] = constraints[categorical_feature].get('rare_threshold', 0.01)
        rules['group_name'] = constraints[categorical_feature].get('rare_group_name', 'Other')
    
    # 2. Standardize values
    df_clean[categorical_feature] = standardize_categorical_values(
        df_clean, categorical_feature, rules)
    
    # 3. Validate against allowed values
    valid_categories = set(constraints[categorical_feature]['valid_categories'])
    if constraints[categorical_feature].get('handle_rare', False):
        valid_categories.add(rules['group_name'])
    
    if rules['missing_strategy'] == 'custom':
        valid_categories.add(rules['missing_value'])
    
    # Find invalid categories after standardization
    actual_categories = set(df_clean[categorical_feature].dropna().unique())
    invalid_categories = actual_categories - valid_categories
    
    if invalid_categories:
        print(f"  Warning: Found {len(invalid_categories)} invalid categories after standardization")
        # Handle invalid categories according to strategy
        if constraints[categorical_feature].get('invalid_strategy', 'replace') == 'replace':
            replacement = constraints[categorical_feature].get('invalid_replacement', 'Other')
            invalid_mask = df_clean[categorical_feature].isin(invalid_categories)
            df_clean.loc[invalid_mask, categorical_feature] = replacement
            print(f"  Replaced {invalid_mask.sum()} invalid values with '{replacement}'")
        else:
            # Convert to NaN and will be handled by missing value imputation
            invalid_mask = df_clean[categorical_feature].isin(invalid_categories)
            df_clean.loc[invalid_mask, categorical_feature] = np.nan
            print(f"  Converted {invalid_mask.sum()} invalid values to NaN")
    
    # 4. Handle any remaining missing values
    missing_after = df_clean[categorical_feature].isna().sum()
    if missing_after > 0:
        print(f"  Handling {missing_after} missing values")
        
        if rules['missing_strategy'] == 'mode':
            mode_value = df_clean[categorical_feature].mode()[0]
            df_clean[categorical_feature] = df_clean[categorical_feature].fillna(mode_value)
            print(f"  Filled missing values with mode: '{mode_value}'")
        elif rules['missing_strategy'] == 'custom':
            custom_value = rules['missing_value']
            df_clean[categorical_feature] = df_clean[categorical_feature].fillna(custom_value)
            print(f"  Filled missing values with custom value: '{custom_value}'")
    
    return df_clean




def analyze_categorical_feature(df, column_name, max_pie_categories=6):
    """
    Analyze a single categorical feature/column, showing detailed statistics for each category 
    and visualizations including missing values.
    Shows top categories ordered from most to least frequent (top to bottom) when there are more than max_pie_categories.
    """
    
    print(f"\n{'-'*50}")
    print(f"Analysis of {column_name}")
    print(f"{'-'*50}")
    
    # Get basic counts
    total_count = len(df)
    null_count = df[column_name].isnull().sum()
    null_percent = (null_count/total_count)*100
    unique_count = df[column_name].nunique() + (1 if null_count > 0 else 0)  # Add 1 for missing values if present
    
    print(f"Total records: {total_count}")
    print(f"Missing values: {null_count} ({null_percent:.2f}%)")
    print(f"Unique categories: {unique_count}")
    
    # Get value counts including nulls
    value_counts = df[column_name].value_counts(dropna=False)
    
    # Detailed information for each category
    print("\nCategory Distribution:")
    print(f"{'Category':<20} {'Count':<10} {'Percentage':<10}")
    print(f"{'-'*40}")
    
    for category, count in value_counts.items():
        category_name = 'Missing' if pd.isna(category) else category
        percentage = (count/total_count)*100
        print(f"{str(category_name)[:20]:<20} {count:<10} {percentage:.2f}%")
    
    # Create visualizations
    plt.figure(figsize=(15, 8))
    
    # Bar chart with missing values
    plt.subplot(121)
    # Replace NaN with 'Missing' for plotting
    plot_data = df[column_name].fillna('Missing')
    
    # Sort values by frequency to make the plot more readable
    plot_counts = plot_data.value_counts().sort_values(ascending=False)
    
    # Use different colors for missing values
    colors = ['#1f77b4' if category != 'Missing' else 'red' for category in plot_counts.index]
    
    # Create the bar chart
    sns.barplot(x=plot_counts.index, y=plot_counts.values, palette=colors)
    plt.title(f"Distribution of {column_name} (including missing values)")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Category")
    plt.ylabel("Count")
    
    # For the second plot: pie chart if few categories, otherwise top categories bar chart
    plt.subplot(122)
    
    if unique_count <= max_pie_categories:
        # Pie chart for few categories
        plt.pie(plot_counts, labels=plot_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.axis('equal')
        plt.title(f"Percentage Distribution of {column_name}")
    else:
        # Alternative plot for many categories: Just show top categories
        top_n = max_pie_categories
        top_categories = plot_counts.head(top_n)
        
        # IMPORTANT: Reverse the order for the horizontal bar chart
        # This makes the most frequent category appear at the top
        top_categories = top_categories.iloc[::-1]
        
        # Colors for the plot (red for Missing)
        bar_colors = ['red' if category == 'Missing' else '#1f77b4' for category in top_categories.index]
        
        # Create horizontal bar chart of just top categories
        y_pos = np.arange(len(top_categories))
        plt.barh(y_pos, top_categories.values, color=bar_colors)
        plt.yticks(y_pos, top_categories.index)
        
        # Add percentage labels
        for i, v in enumerate(top_categories.values):
            percentage = (v/total_count)*100
            plt.text(v + (total_count*0.01), i, f"{percentage:.1f}%", va='center')
        
        plt.title(f"Top {top_n} Categories (out of {unique_count})")
        plt.xlabel("Count")
    
    plt.tight_layout()
    plt.show()
    

    