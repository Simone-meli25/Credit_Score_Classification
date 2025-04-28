import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class ImputationEvaluator:
    """
    Class for evaluating and selecting the best imputation method for each feature.
    
    This class implements a rigorous methodology to:
    1. Create artificial missing values in a holdout set
    2. Test multiple imputation methods
    3. Evaluate each method using multiple metrics
    4. Select the best method for each feature
    5. Apply the best method to the real data
    """
    
    def __init__(self, seed=42):
        """
        Initialize the evaluator with a random seed for reproducibility.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Define imputation methods to test
        self.imputation_methods = {
            'mean': self._impute_mean,
            'median': self._impute_median,
            'mode': self._impute_mode,
            'knn_3': lambda df, col: self._impute_knn(df, col, 3),
            #'knn_5': lambda df, col: self._impute_knn(df, col, 5),
            #'knn_7': lambda df, col: self._impute_knn(df, col, 7),
            #'knn_9': lambda df, col: self._impute_knn(df, col, 9),
            #'mice': self._impute_mice
        }
        
        # Store results
        self.evaluation_results = {}
        self.best_methods = {}
    
    def _create_holdout_mask(self, df, column, holdout_fraction=0.2):
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
    
    def _impute_mean(self, df, column):
        """Impute missing values with column mean."""
        mean_value = df[column].mean()
        df_imputed = df.copy()
        df_imputed[column] = df_imputed[column].fillna(mean_value)
        return df_imputed
    
    def _impute_median(self, df, column):
        """Impute missing values with column median."""
        median_value = df[column].median()
        df_imputed = df.copy()
        df_imputed[column] = df_imputed[column].fillna(median_value)
        return df_imputed
    
    def _impute_mode(self, df, column):
        """Impute missing values with column mode."""
        mode_value = df[column].mode()[0]
        df_imputed = df.copy()
        df_imputed[column] = df_imputed[column].fillna(mode_value)
        return df_imputed
    
    def _impute_knn(self, df, column, n_neighbors=5):
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
    
    def _impute_mice(self, df, column):
        """
        Impute missing values using MICE (Multiple Imputation by Chained Equations).
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column to impute
            
        Returns:
            pd.DataFrame: Dataframe with imputed values
        """
        df_imputed = df.copy()
        
        # Get numeric columns for MICE
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        df_numeric = df[numeric_cols].copy()
        
        # Fill any other missing values temporarily for MICE to work
        for col in numeric_cols:
            if col != column and df_numeric[col].isna().any():
                df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())
        
        # Apply MICE imputation
        imputer = IterativeImputer(
            max_iter=10,
            random_state=self.seed,
            min_value=0 if column.endswith(('_Debt', '_Income', '_Salary', 'Monthly', '_Ratio', '_Rate')) else None
        )
        
        df_imputed_numeric = pd.DataFrame(
            imputer.fit_transform(df_numeric),
            columns=numeric_cols,
            index=df_numeric.index
        )
        
        # Copy only the target column's imputed values
        df_imputed[column] = df_imputed_numeric[column]
        
        # If it's a count variable, round to integers
        if column.startswith('Num_'):
            df_imputed[column] = np.round(df_imputed[column]).astype('Int64')
        
        return df_imputed
    
    def evaluate_methods_for_column(self, df, column, holdout_fraction=0.2):
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
        holdout_mask, holdout_values = self._create_holdout_mask(df, column, holdout_fraction)
        
        # Create a copy with artificial missing values
        df_with_artificial_missing = df.copy()
        df_with_artificial_missing.loc[holdout_mask, column] = np.nan
        
        # Calculate original statistics for comparison
        original_stats = df[column].describe()
        
        # Evaluate each imputation method
        results = {}
        
        for method_name, impute_func in self.imputation_methods.items():
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
    
    def evaluate_all_numeric_columns(self, df, holdout_fraction=0.2):
        """
        Evaluate imputation methods for all numeric columns with missing values.
        
        Args:
            df (pd.DataFrame): Input dataframe
            holdout_fraction (float): Fraction of data to hold out for evaluation
            
        Returns:
            dict: Best imputation method for each column
        """
        # Get numeric columns with missing values
        numeric_cols = df.select_dtypes(include=['number']).columns
        missing_cols = [col for col in numeric_cols if df[col].isna().any()]
        
        print(f"Evaluating imputation methods for {len(missing_cols)} numeric columns")
        
        for col in missing_cols:
            self.evaluate_methods_for_column(df, col, holdout_fraction)
        
        # Summarize results
        print("\nImputation Method Evaluation Summary:")
        for col, method in self.best_methods.items():
            if method:
                score = self.evaluation_results[col][method]['combined_score']
                print(f"  {col}: {method} (score: {score:.4f})")
            else:
                print(f"  {col}: No valid method found")
        
        return self.best_methods
    
    def impute_with_best_methods(self, df):
        """
        Impute the dataframe using the best method for each column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with all columns imputed using their best method
        """
        df_imputed = df.copy()
        
        for col, method in self.best_methods.items():
            if not method:
                print(f"Skipping {col} - no valid imputation method found")
                continue
                
            print(f"Imputing {col} using {method}")
            
            # Get the imputation function
            impute_func = self.imputation_methods[method]
            
            # Apply imputation to the full dataset
            df_imputed = impute_func(df_imputed, col)
        
        return df_imputed
    
    def visualize_method_comparison(self, df, column):
        """
        Visualize the comparison of different imputation methods for a column.
        
        Args:
            df (pd.DataFrame): Original dataframe
            column (str): Column to visualize
        """
        if column not in self.evaluation_results:
            print(f"No evaluation results for column '{column}'")
            return
            
        results = self.evaluation_results[column]
        valid_methods = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_methods:
            print(f"No valid imputation methods for column '{column}'")
            return
        
        # Create holdout data for visualization
        holdout_mask, _ = self._create_holdout_mask(df, column, 0.2)
        df_missing = df.copy()
        df_missing.loc[holdout_mask, column] = np.nan
        
        # Plot comparison
        n_methods = len(valid_methods)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        plt.figure(figsize=(18, n_rows * 5))
        
        # First, plot the original distribution
        ax_orig = plt.subplot(n_rows + 1, n_cols, 1)
        sns.histplot(df[column].dropna(), kde=True, color='blue', ax=ax_orig)
        ax_orig.set_title(f'Original Distribution\n{column}')
        
        # Then plot each imputation method
        for i, (method_name, metrics) in enumerate(valid_methods.items(), 1):
            ax = plt.subplot(n_rows + 1, n_cols, i + n_cols)
            
            # Apply imputation
            df_imputed = self.imputation_methods[method_name](df_missing, column)
            
            # Plot histogram
            sns.histplot(df_imputed[column], kde=True, color='green', ax=ax)
            
            # Add metrics to title
            title = f"{method_name}\nRMSE: {metrics['rmse']:.4f}, Dist Score: {metrics['distribution_score']:.2f}%"
            ax.set_title(title)
            
        plt.tight_layout()
        plt.show()
        
        # Create a comparison bar chart of metrics
        plt.figure(figsize=(12, 6))
        
        # Extract RMSE and distribution scores
        methods = list(valid_methods.keys())
        rmse_scores = [valid_methods[m]['rmse'] for m in methods]
        dist_scores = [valid_methods[m]['distribution_score'] for m in methods]
        combined_scores = [valid_methods[m]['combined_score'] for m in methods]
        
        # Create bar positions
        x = np.arange(len(methods))
        width = 0.25
        
        # Create bars
        plt.bar(x - width, rmse_scores, width, label='RMSE')
        plt.bar(x, dist_scores, width, label='Distribution Score (%)')
        plt.bar(x + width, combined_scores, width, label='Combined Score')
        
        # Add labels and legend
        plt.xlabel('Imputation Method')
        plt.ylabel('Score (lower is better)')
        plt.title(f'Imputation Method Comparison for {column}')
        plt.xticks(x, methods, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.show()


def auto_impute_numeric_features(df, holdout_fraction=0.2, seed=42):
    """
    Automatically determine and apply the best imputation method for each numeric feature.
    
    Args:
        df (pd.DataFrame): Input dataframe
        holdout_fraction (float): Fraction of data to hold out for evaluation
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (imputed_df, evaluator)
    """
    # Initialize evaluator
    evaluator = ImputationEvaluator(seed=seed)
    
    # Evaluate methods for numeric columns
    evaluator.evaluate_all_numeric_columns(df, holdout_fraction)
    
    # Apply best methods
    df_imputed = evaluator.impute_with_best_methods(df)
    
    return df_imputed, evaluator


def auto_impute_categorical_features(df, default_method='mode'):
    """
    Impute missing values in categorical features using the specified method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        default_method (str): Default imputation method ('mode', 'new_category')
        
    Returns:
        pd.DataFrame: Dataframe with imputed values
    """
    df_imputed = df.copy()
    
    # Get categorical columns with missing values
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    missing_cols = [col for col in categorical_cols if df[col].isna().any()]
    
    if not missing_cols:
        print("No categorical columns with missing values to impute")
        return df_imputed
    
    print(f"Imputing {len(missing_cols)} categorical columns using '{default_method}'")
    
    # Process each column
    for col in missing_cols:
        missing_count = df_imputed[col].isna().sum()
        
        print(f"  Imputing '{col}' ({missing_count} missing values)")
        
        if default_method == 'mode':
            # Use most frequent value
            mode_value = df_imputed[col].mode()[0]
            df_imputed[col] = df_imputed[col].fillna(mode_value)
            print(f"    Using most frequent value: {mode_value}")
            
        elif default_method == 'new_category':
            # Create a new category for missing values
            df_imputed[col] = df_imputed[col].fillna('Missing')
            print(f"    Using 'Missing' as a new category")
            
        else:
            raise ValueError(f"Unknown categorical imputation method: {default_method}")
    
    return df_imputed



