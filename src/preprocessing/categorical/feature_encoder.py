import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEncoder(BaseEstimator, TransformerMixin):
    """
    A transformer for encoding categorical features with different strategies.
    
    Parameters:
    -----------
    encoding_map : dict
        Dictionary mapping feature names to encoding strategies.
        For fixed strategies: feature_name: 'strategy_name'
        For multiple options: feature_name: ('default_strategy', ['alternative1', 'alternative2'])
        
    encoding_params : dict, optional
        Dictionary with specific parameters for each feature encoding if needed.
        
    hyperparams : dict, optional
        Dictionary specifying which strategy to use for features with multiple options.
        Format: {'feature_name': 'chosen_strategy'}
    
    Attributes:
    -----------
    encoders_ : dict
        Fitted encoding transformations for each feature
    """
    def __init__(self, encoding_map, encoding_params=None, hyperparams=None):
        self.encoding_map = encoding_map
        self.encoding_params = encoding_params or {}
        self.hyperparams = hyperparams or {} # will be modified during hyperparameter tuning by sklearn's parameter setting mechanism to test different encoding strategies
        
    def fit(self, X, y=None):
        """
        Learn the required statistics for each encoding strategy.
        
        Parameters:
        -----------
        X : pandas DataFrame
            The input features to fit
        y : array-like, default=None
            Target values for target encoding
            
        Returns:
        --------
        self : object
            Returns self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
            
        # Initialize storage for fitted encoders/transformations
        self.encoders_ = {}
        
        # Process each feature in the encoding map
        for feature, strategy_info in self.encoding_map.items():
            if feature not in X.columns:
                raise ValueError(f"Feature {feature} present in encoding_map, but not found in DataFrame columns")

            # Determine the strategy to use
            if isinstance(strategy_info, tuple):
                # This feature has multiple strategy options
                default_strategy = strategy_info[0]
                # Check if a specific strategy was selected via hyperparams
                strategy = self.hyperparams.get(feature, default_strategy)
            else:
                # This feature has a fixed strategy
                strategy = strategy_info
                
            # Get feature-specific parameters if any
            params = self.encoding_params.get(feature, {})
            
            # Fit the appropriate encoder based on strategy
            if strategy == 'ordinal':
                # For ordinal encoding, we need the mapping
                if 'order_map' in params:
                    self.encoders_[feature] = params['order_map']
                else:
                    # Create default mapping if not provided
                    categories = X[feature].dropna().unique()
                    self.encoders_[feature] = {cat: i for i, cat in enumerate(categories)}
                    
            elif strategy == 'one-hot':
                # For one-hot, just store the unique categories
                self.encoders_[feature] = X[feature].dropna().unique()
                
            elif strategy == 'frequency':
                # For frequency encoding, compute value counts
                self.encoders_[feature] = X[feature].value_counts(normalize=True).to_dict()
                
            elif strategy == 'target':
                # For target encoding, compute mean target value per category
                if y is None:
                    raise ValueError("Target values (y) are required for target encoding")
                self.encoders_[feature] = X.groupby(feature)[y.name].mean().to_dict() if hasattr(y, 'name') else \
                                         pd.Series(y).groupby(X[feature]).mean().to_dict()
                
            elif strategy == 'binary':
                # For binary encoding, create a simple 0/1 mapping
                categories = X[feature].dropna().unique()
                if len(categories) != 2:
                    raise ValueError(f"Binary encoding expects 2 categories, got {len(categories)} for {feature}")
                self.encoders_[feature] = {categories[0]: 0, categories[1]: 1}
            
            # Add other encoding strategies as needed
                
        return self
        
    def transform(self, X):
        """
        Apply the encoding transformations to the data.
        
        Parameters:
        -----------
        X : pandas DataFrame
            The input features to transform
            
        Returns:
        --------
        pandas DataFrame
            Transformed data with encoded features
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
            
        # Make a copy to avoid modifying the original data
        X_transformed = X.copy()
        
        # Process each feature in the encoding map
        for feature, strategy_info in self.encoding_map.items():
            if feature not in X.columns or feature not in self.encoders_:
                raise ValueError(f"Feature {feature} from encoding_map not found in DataFrame columns or encoders")
                
            # Determine the strategy to use
            if isinstance(strategy_info, tuple):
                # This feature has multiple strategy options
                default_strategy = strategy_info[0]
                # Check if a specific strategy was selected via hyperparams
                strategy = self.hyperparams.get(feature, default_strategy)
            else:
                # This feature has a fixed strategy
                strategy = strategy_info
                
            # Apply the appropriate transformation based on strategy
            if strategy == 'ordinal':
                # Map values using the encoder dictionary
                encoder = self.encoders_[feature]
                # Handle unseen categories with a default value
                X_transformed[feature] = X[feature].map(encoder).fillna(-1)
                
            elif strategy == 'one-hot':
                # Create dummy variables
                dummies = pd.get_dummies(X[feature], prefix=feature)
                # Add the dummies to the dataframe
                X_transformed = pd.concat([X_transformed, dummies], axis=1)
                # Drop the original column
                X_transformed = X_transformed.drop(feature, axis=1)
                
            elif strategy == 'frequency':
                # Map values to their frequencies
                encoder = self.encoders_[feature]
                # Handle unseen categories with a default value (0)
                X_transformed[feature] = X[feature].map(encoder).fillna(0)
                
            elif strategy == 'target':
                # Map values to target means
                encoder = self.encoders_[feature]
                # Handle unseen categories with global mean
                global_mean = sum(encoder.values()) / len(encoder) if encoder else 0
                X_transformed[feature] = X[feature].map(encoder).fillna(global_mean)
                
            elif strategy == 'binary':
                # Map to binary values
                encoder = self.encoders_[feature]
                # Handle unseen categories with -1
                X_transformed[feature] = X[feature].map(encoder).fillna(-1)
            
            # Add other encoding strategies as needed
                
        return X_transformed
    
    def set_hyperparams(self, hyperparams):
        """
        Update the hyperparameters for features with multiple encoding options.
        
        Parameters:
        -----------
        hyperparams : dict
            Dictionary specifying which strategy to use for features with multiple options.
            Format: {'feature_name': 'chosen_strategy'}
            
        Returns:
        --------
        self : object
            Returns self
        """
        self.hyperparams = hyperparams
        return self