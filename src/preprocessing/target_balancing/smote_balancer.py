# src/preprocessing/balancing/smote_balancer.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE

class SMOTEBalancer(BaseEstimator, TransformerMixin):
    """
    A transformer for balancing imbalanced classes using SMOTE.
    
    Parameters:
    -----------
    sampling_strategy : str or dict or float, default='auto'
        Sampling strategy to use. 'auto' uses all classes as minority.
        'minority' uses only the minority class.
        'not majority' uses all classes except the majority class.
        'all' generates samples for all classes except the majority class.
        float - ratio of samples in each class to samples in majority class.
        dict - keys are classes, values are number of samples for each class.
    
    k_neighbors : int, default=5
        Number of nearest neighbors to use for SMOTE.
        
    random_state : int, default=None
        Random state for reproducibility.
    
    Attributes:
    -----------
    smote_ : SMOTE
        The fitted SMOTE instance.
    """
    def __init__(self, sampling_strategy='auto', k_neighbors=5, random_state=None):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Initialize the SMOTE balancer. This doesn't actually do balancing yet.
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            The input features to fit
        y : array-like
            Target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        self.smote_ = SMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state
        )
        return self
    
    def transform(self, X, y=None):
        """
        Apply SMOTE to the input data.
        This method doesn't actually transform X, but is implemented to conform
        to the sklearn Transformer interface. In practice, the resampling is done
        outside of the pipeline in most cases.
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            The input features to transform
        y : array-like, default=None
            Target values
            
        Returns:
        --------
        X : pandas DataFrame or numpy array
            The input features (unchanged)
        """
        # In a standard sklearn pipeline, transform doesn't modify X
        # SMOTE resampling will be done in a special fit_resample method
        return X
    
    def fit_resample(self, X, y):
        """
        Fit the SMOTE model and balance the dataset.
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            The input features to resample
        y : array-like
            Target values
            
        Returns:
        --------
        X_resampled : pandas DataFrame or numpy array
            The resampled features
        y_resampled : array-like
            The resampled target values
        """
        self.fit(X, y)
        
        # Remember if X was a DataFrame
        was_dataframe = isinstance(X, pd.DataFrame)
        column_names = X.columns if was_dataframe else None
        
        # Apply SMOTE
        X_resampled, y_resampled = self.smote_.fit_resample(X, y)
        
        # Convert back to DataFrame if input was a DataFrame
        if was_dataframe:
            X_resampled = pd.DataFrame(X_resampled, columns=column_names)
        
        return X_resampled, y_resampled