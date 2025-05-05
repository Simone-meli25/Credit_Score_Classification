import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    A transformer for handling outliers in numerical data.
    
    Parameters:
    -----------
    strategy : str, default='none'
        Method to handle outliers:
        - 'none': Keep all data as is
        - 'clip': Clip values beyond the z-score threshold
    
    z_thresh : float, default=3.0
        Z-score threshold to identify outliers. Values with absolute 
        z-scores greater than this are considered outliers.
    
    Attributes:
    -----------
    means_ : pandas Series
        Column means (only stored when strategy='clip')
    
    stds_ : pandas Series
        Column standard deviations (only stored when strategy='clip')
    """
    def __init__(self, strategy='none', z_thresh=3.0):
        self.strategy = strategy  # choices: 'none', 'clip', 'remove'
        self.z_thresh = z_thresh  # numeric threshold
    
    def fit(self, X, y=None):
        """
        Learn the required statistics from the training data.
        
        Parameters:
        -----------
        X : pandas DataFrame
            The input features to fit
        y : array-like, default=None
            Ignored, exists for compatibility
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Validate that X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
            
        # Validate strategy
        valid_strategies = ['none', 'clip']
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
        
        # Only select numeric columns for outlier handling
        numeric_cols = X.select_dtypes(include=['number']).columns
        self.numeric_cols_ = numeric_cols  # Store for later use in transform
            
        # Store column means and stds if we'll use them for clipping
        if self.strategy == 'clip':
            if len(numeric_cols) > 0:
                self.means_ = X[numeric_cols].mean()
                self.stds_ = X[numeric_cols].std()
                
                # Check for constant columns to avoid division by zero
                zero_std_cols = self.stds_[self.stds_ == 0].index.tolist()
                if zero_std_cols:
                    raise ValueError(f"Found constant columns: {zero_std_cols}. "
                                    "These columns have zero standard deviation.")
            
            else:
                raise ValueError("No numeric columns found for outlier handling")
                
        return self
    
    def transform(self, X):
        """
        Apply the outlier handling strategy to the data.
        
        Parameters:
        -----------
        X : pandas DataFrame
            The input features to transform
            
        Returns:
        --------
        pandas DataFrame
            Transformed data with outliers handled according to strategy
        """
        # Validate that X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
            
        # Make a copy to avoid modifying the original data
        X_transformed = X.copy()
        
        # Get numeric columns (either from fit or find them now)
        numeric_cols = getattr(self, 'numeric_cols_', X.select_dtypes(include=['number']).columns)
        
        if self.strategy == 'none' or len(numeric_cols) == 0:
            # Do nothing, return data as is
            return X_transformed
            
        elif self.strategy == 'clip':
            # Use stored means and stds from fit method
            # Calculate z-scores for each value (only for numeric columns)
            z_scores = (X_transformed[numeric_cols] - self.means_) / self.stds_
            
            # Identify outliers
            outlier_mask = abs(z_scores) > self.z_thresh
            
            # For each column with outliers
            for col in outlier_mask.columns[outlier_mask.any()]:
                # Get column mask
                col_mask = outlier_mask[col]
                
                # Clip the outliers to the threshold
                upper_bound = self.means_[col] + (self.z_thresh * self.stds_[col])
                lower_bound = self.means_[col] - (self.z_thresh * self.stds_[col])
                
                # Replace values outside bounds
                X_transformed.loc[col_mask & (X_transformed[col] > upper_bound), col] = upper_bound
                X_transformed.loc[col_mask & (X_transformed[col] < lower_bound), col] = lower_bound
                
        '''
        elif self.strategy == 'remove':
            # Calculate z-scores for the numeric data
            z_scores = (X_transformed[numeric_cols] - X_transformed[numeric_cols].mean()) / X_transformed[numeric_cols].std()
            
            # Identify rows to keep (those without outliers in any column)
            rows_to_keep = (abs(z_scores) <= self.z_thresh).all(axis=1)
            
            # Keep only non-outlier rows
            X_transformed = X_transformed.loc[rows_to_keep]

        '''
            
        return X_transformed