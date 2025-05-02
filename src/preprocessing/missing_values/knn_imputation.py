import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error



def impute_knn(df, column_to_impute, n_neighbors=5):
    """
    Impute missing values using KNN imputation.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Single column to impute
        n_neighbors (int): Number of neighbors for KNN
        
    Returns:
        pd.DataFrame: Dataframe with imputed values
    """
    df_copy = df.copy()
    
    # Store original dtype to restore it later
    original_dtype = df_copy[column_to_impute].dtype

        # Check if target column is all missing
    if df_copy[column_to_impute].isna().all():
        raise ValueError(f"Warning: All values in column '{column_to_impute}' are missing")
    
    # Get numeric columns for KNN
    num_cols = df_copy.select_dtypes(include=['number']).columns.tolist()

    # 2) Pre-fill **all** numeric missing with median (including target)
    medians = df_copy[num_cols].median()
    df_filled = df_copy[num_cols].fillna(medians)
    
    # Standardize features for KNN
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_filled),
        columns=num_cols,
        index=df_copy.index
    )

    # Re-mask the target column to NaN (so only it gets imputed)
    missing_mask = df_copy[column_to_impute].isna()
    df_scaled.loc[missing_mask, column_to_impute] = np.nan
    
    # Apply KNN imputation on the scaled data
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed_scaled = pd.DataFrame(
        imputer.fit_transform(df_scaled),
        columns=num_cols,
        index=df_copy.index
    )
    
    # Inverse transform to get original scale
    df_imputed_original_scale = pd.DataFrame(
        scaler.inverse_transform(df_imputed_scaled),
        columns=num_cols,
        index=df_copy.index
    )
    
    # Copy only the target column's imputed values
    df_copy[column_to_impute] = df_imputed_original_scale[column_to_impute]
    
    # Restore original dtype if it was integer
    if pd.api.types.is_integer_dtype(original_dtype):
        df_copy[column_to_impute] = df_copy[column_to_impute].round().astype(original_dtype)
    
    return df_copy


def create_holdout_mask(df, column, holdout_fraction=0.2):
    """
    Create a mask for artificial missing values in a holdout set.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to create holdout for
        holdout_fraction (float): Fraction of non-missing values to hold out
        
    Returns:
        tuple: (holdout_mask, holdout_values)
    """

    np.random.seed(42)   # for reproducible hold-out

    # Only consider non-missing values for holdout
    non_missing_mask = ~df[column].isna()
    non_missing_indices = np.where(non_missing_mask)[0]
    
    # Calculate number of values to hold out
    n_holdout = int(len(non_missing_indices) * holdout_fraction)
    
    # Randomly select indices to hold out
    holdout_indices = np.random.choice(
        non_missing_indices, 
        size=n_holdout, 
        replace=False
    )
    
    # Create mask where True indicates values to hold out
    holdout_mask = np.zeros(len(df), dtype=bool)
    holdout_mask[holdout_indices] = True
    
    # Get the actual values being held out
    holdout_values = df.loc[holdout_mask, column].copy()
    
    return holdout_mask, holdout_values


def plot_distribution_comparison(original_df, imputed_df, column, rmse=None, dist_score=None):
    """
    Plot the distribution of original and imputed values for a column.
    
    Args:
        original_df (pd.DataFrame): Original dataframe
        imputed_df (pd.DataFrame): Dataframe with imputed values
        column (str): Column name to plot
        rmse (float, optional): Root Mean Squared Error to display
        dist_score (float, optional): Distribution score to display
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot original distribution
    sns.histplot(original_df[column], kde=True, ax=ax1, color='blue', alpha=0.7)
    ax1.set_title(f"Original Distribution\n{column}")
    ax1.set_xlabel(column)
    ax1.set_ylabel("Count")
    
    # Plot imputed distribution
    title = f"knn_{k}" if 'k' in globals() else "KNN Imputation"
    if rmse is not None and dist_score is not None:
        title += f"\nRMSE: {rmse:.4f}, Dist Score: {dist_score:.2f}%"
    
    sns.histplot(imputed_df[column], kde=True, ax=ax2, color='green', alpha=0.7)
    ax2.set_title(title)
    ax2.set_xlabel(column)
    ax2.set_ylabel("Count")
    
    plt.tight_layout()
    plt.show()


def apply_and_evaluate_knn_imputation(df, columns, holdout_fraction=0.2, k=3):
    """
    Evaluate knn imputation method for specific columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of columns to evaluate KNN imputation method for
        holdout_fraction (float): Fraction of data to hold out for evaluation
        k (int): Number of neighbors for KNN

    Returns:
        tuple: (imputed_df, results_dict) - DataFrame with all imputed values and dictionary of results per column
    """
    df_copy = df.copy()
    results_dict = {}  # Store results for all columns
    
    # Start with a copy of the original data as the base for imputation
    df_imputed = df_copy.copy()

    for column in columns:
        print(f"\nEvaluating KNN imputation (k = {k}) for column '{column}'\n")
        
        # Create holdout set
        holdout_mask, holdout_values = create_holdout_mask(df_copy, column, holdout_fraction)
        
        # Create a copy with artificial missing values
        df_with_artificial_missing = df_copy.copy()  # Make a proper copy
        df_with_artificial_missing.loc[holdout_mask, column] = np.nan
            
        try:
            # Apply imputation
            column_imputed_df = impute_knn(df_with_artificial_missing, column, k)
            
            # Update the imputed values in the final dataframe
            df_imputed[column] = column_imputed_df[column]
            
            # Calculate metrics
            imputed_values = column_imputed_df.loc[holdout_mask, column]
            
            # Root mean squared error
            rmse = np.sqrt(mean_squared_error(holdout_values, imputed_values))

            # Calculate original statistics for comparison
            original_stats = df[column].describe()
            
            # Distribution statistics
            imputed_stats = column_imputed_df[column].describe()
            
            # Calculate difference in key statistics
            delta_mean = abs((imputed_stats['mean'] - original_stats['mean']) / original_stats['mean'] * 100)
            delta_std = abs((imputed_stats['std'] - original_stats['std']) / original_stats['std'] * 100)
            delta_25 = abs((imputed_stats['25%'] - original_stats['25%']) / original_stats['25%'] * 100)
            delta_75 = abs((imputed_stats['75%'] - original_stats['75%']) / original_stats['75%'] * 100)
            
            # Combined distribution fidelity score (lower is better)
            dist_score = (delta_mean + delta_std + delta_25 + delta_75) / 4
            
            # Store results for this column
            results_dict[column] = {
                'rmse': rmse,
                'delta_mean': delta_mean,
                'delta_std': delta_std,
                'delta_25': delta_25,
                'delta_75': delta_75,
                'distribution_score': dist_score,
                'combined_score': rmse * (1 + dist_score/100)  # Weighted combined score
            }
            
            # Plot distribution comparison
            plot_distribution_comparison(
                df_copy, 
                column_imputed_df, 
                column,
                rmse=rmse,
                dist_score=dist_score
            )
                        
        except Exception as e:
            print(f"    Error with KNN imputation for {column}: {str(e)}")
            results_dict[column] = {'error': str(e)}

        if not column in results_dict:
            raise ValueError(f"No valid results for column {column}")

    return df_imputed, results_dict



        
def apply_and_evaluate_knn_for_all_numeric_columns(df, missing_cols, holdout_fraction=0.2, k=3):
    """
    Evaluate KNN imputation method for all numeric columns with missing values.
    
    Args:
        df (pd.DataFrame): Input dataframe
        holdout_fraction (float): Fraction of data to hold out for evaluation
        
    Returns:
        dict: Best imputation method for each column
    """
    df_copy = df.copy()
                
    print(f"Evaluating KNN imputation method for {len(missing_cols)} numeric columns")
    
    df_imputed, results_for_column = apply_and_evaluate_knn_imputation(df_copy, missing_cols, holdout_fraction, k)

    return df_imputed, results_for_column

        
