
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error



def impute_knn(df, column, n_neighbors=5):
    """
    Impute missing values using KNN imputation.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column to impute
        n_neighbors (int): Number of neighbors for KNN
        
    Returns:
        pd.DataFrame: Dataframe with imputed values
    """
    df_imputed = df.copy()
    
    # Get numeric columns for KNN
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Create a standardized copy for KNN
    df_numeric = df[numeric_cols].copy()
    
    # Fill any other missing values temporarily for KNN to work
    for col in numeric_cols:
        if col != column and df_numeric[col].isna().any():
            df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())
    
    # Standardize features for KNN
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_numeric),
        columns=numeric_cols,
        index=df_numeric.index
    )
    
    # Apply KNN imputation
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed_scaled = pd.DataFrame(
        imputer.fit_transform(df_scaled),
        columns=numeric_cols,
        index=df_scaled.index
    )
    
    # Inverse transform to get original scale
    df_imputed_numeric = pd.DataFrame(
        scaler.inverse_transform(df_imputed_scaled),
        columns=numeric_cols,
        index=df_imputed_scaled.index
    )
    
    # Copy only the target column's imputed values
    df_imputed[column] = df_imputed_numeric[column]
    
    return df_imputed


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


def evaluate_knn_imputation(df, column, holdout_fraction=0.2):
        """
        Evaluate all imputation methods for a specific column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column to evaluate imputation methods for
            holdout_fraction (float): Fraction of data to hold out for evaluation
            
        Returns:
            dict: Evaluation metrics for each method
        """
        print(f"\nEvaluating imputation methods for column '{column}'")
        
        # Create holdout set
        holdout_mask, holdout_values = create_holdout_mask(df, column, holdout_fraction)
        
        # Create a copy with artificial missing values
        df_with_artificial_missing = df.copy()
        df_with_artificial_missing.loc[holdout_mask, column] = np.nan
        
        # Calculate original statistics for comparison
        original_stats = df[column].describe()
        
        # Evaluate each imputation method
        results = {}
        
        for method_name, impute_func in imputation_methods.items():
            print(f"  Testing {method_name}...")
            
            try:
                # Apply imputation
                df_imputed = impute_func(df_with_artificial_missing, column)
                
                # Calculate metrics
                imputed_values = df_imputed.loc[holdout_mask, column]
                
                # Root mean squared error
                rmse = np.sqrt(mean_squared_error(holdout_values, imputed_values))
                
                # Distribution statistics
                imputed_stats = df_imputed[column].describe()
                
                # Calculate difference in key statistics
                delta_mean = abs((imputed_stats['mean'] - original_stats['mean']) / original_stats['mean'] * 100)
                delta_std = abs((imputed_stats['std'] - original_stats['std']) / original_stats['std'] * 100)
                delta_25 = abs((imputed_stats['25%'] - original_stats['25%']) / original_stats['25%'] * 100)
                delta_75 = abs((imputed_stats['75%'] - original_stats['75%']) / original_stats['75%'] * 100)
                
                # Combined distribution fidelity score (lower is better)
                dist_score = (delta_mean + delta_std + delta_25 + delta_75) / 4
                
                # Store results
                results[method_name] = {
                    'rmse': rmse,
                    'delta_mean': delta_mean,
                    'delta_std': delta_std,
                    'delta_25': delta_25,
                    'delta_75': delta_75,
                    'distribution_score': dist_score,
                    'combined_score': rmse * (1 + dist_score/100)  # Weighted combined score
                }
                
                print(f"    RMSE: {rmse:.4f}, Distribution Score: {dist_score:.2f}%")
                
            except Exception as e:
                print(f"    Error with {method_name}: {str(e)}")
                results[method_name] = {'error': str(e)}
        
        # Store results for this column
        self.evaluation_results[column] = results
        
        # Determine best method (lowest combined score)
        valid_methods = {k: v for k, v in results.items() if 'error' not in v}
        if valid_methods:
            best_method = min(valid_methods.items(), key=lambda x: x[1]['combined_score'])[0]
            self.best_methods[column] = best_method
            print(f"  Best method for '{column}': {best_method}")
        else:
            print(f"  No valid imputation methods for '{column}'")
            self.best_methods[column] = None
        
        return results