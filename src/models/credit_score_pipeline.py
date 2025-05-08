# src/models/credit_score_pipeline.py
import numpy as np
import pandas as pd
import yaml
import os
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom components
from src.preprocessing.outliers.outliers_handler import OutlierHandler
from src.preprocessing.categorical.feature_encoder import FeatureEncoder
from imblearn.over_sampling import SMOTE

class CreditScorePipeline:
    """
    End-to-end pipeline for credit score classification using XGBoost.
    
    This class handles:
    1. Data splitting (train/validation/test)
    2. Preprocessing with hyperparameters (outliers, encoding)
    3. Class balancing (SMOTE)
    4. Hyperparameter tuning optimization (RandomizedSearchCV)
    5. Model evaluation
    
    Parameters:
    -----------
    config_path : str, default='config.yaml'
        Path to configuration file
        
    random_state : int, default=42
        Random seed for reproducibility
    """
    def __init__(self, config_path='config.yaml', random_state=42):
        self.config_path = config_path
        self.random_state = random_state
        self.config = self._load_config()
        
        # Initialize empty pipeline attributes
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.best_model = None
        self.feature_importances = None
        
        # Initialize empty encoding maps
        self.encoding_map = None
        self.encoding_params = None
        self.param_grid = None

    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
        

    def split_data(self, X, y, test_size=0.2, val_size=0.2, custom_split=False, customer_id_col='Customer_ID'):
        """
        Split the data into train, validation, and test sets.
        
        Parameters:
        -----------
        X : pandas DataFrame
            Input features
            
        y : pandas Series or array-like
            Target values
            
        test_size : float, default=0.2
            Proportion of data to use as test set
            
        val_size : float, default=0.2
            Proportion of remaining data to use as validation set
            
        custom_split : bool, default=False
            If True, ensures all records from the same customer stay together
            If False, performs random split of individual records
            
        customer_id_col : str, default='Customer_ID'
            Name of the column containing customer IDs (only used if custom_split=True)
        """
        if custom_split:
            self._split_by_customer(X, y, test_size, val_size, customer_id_col)
        else:
            self._split_random(X, y, test_size, val_size)
        
        # Create encoding map and param grid after data split
        self.encoding_map, self.encoding_params = self._create_encoding_map()
        self.param_grid = self._create_param_grid()

    def _split_random(self, X, y, test_size, val_size):
        """Perform random split of individual records."""
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        # Second split: separate train and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=self.random_state,
            stratify=y_temp
        )
        
        print(f"Random split into train ({len(self.X_train)} records), "
              f"validation ({len(self.X_val)} records), and "
              f"test ({len(self.X_test)} records) sets.")

    def _split_by_customer(self, X, y, test_size, val_size, customer_id_col):
        """Split data ensuring all records from the same customer stay together."""
        # Get unique customer IDs
        unique_customers = X[customer_id_col].unique()
        
        # Split customer IDs into train, validation, and test sets
        customers_temp, test_customers = train_test_split(
            unique_customers, 
            test_size=test_size, 
            random_state=self.random_state
        )
        
        train_customers, val_customers = train_test_split(
            customers_temp, 
            test_size=val_size, 
            random_state=self.random_state
        )
        
        # Create masks for each set
        train_mask = X[customer_id_col].isin(train_customers)
        val_mask = X[customer_id_col].isin(val_customers)
        test_mask = X[customer_id_col].isin(test_customers)
        
        # Split the data using the masks
        self.X_train = X[train_mask]
        self.X_val = X[val_mask]
        self.X_test = X[test_mask]
        
        self.y_train = y[train_mask]
        self.y_val = y[val_mask]
        self.y_test = y[test_mask]

        # Drop the customer_id column from the train, validation and test sets
        self.X_train = self.X_train.drop(columns=[customer_id_col])
        self.X_val = self.X_val.drop(columns=[customer_id_col])
        self.X_test = self.X_test.drop(columns=[customer_id_col])
        
        print(f"Feature-based split using {customer_id_col} (then dropped) into train ({len(train_customers)} customers, {len(self.X_train)} records), "
              f"validation ({len(val_customers)} customers, {len(self.X_val)} records), and "
              f"test ({len(test_customers)} customers, {len(self.X_test)} records) sets.")

    def _create_encoding_map(self):
        """Create encoding map and params from config."""
        encoding_map = {}
        encoding_params = {}
        
        # Get features that are actually present in the data
        available_features = set(self.X_train.columns)
        
        # settings contains the feature's strategy, alternatives (in case of more than one strategy), 
        # and params (in case of ordinal strategy)
        for feature, settings in self.config['preprocessing']['encoding']['features'].items():
            # Skip features that are not in the data
            if feature not in available_features:
                print(f"Warning: Feature '{feature}' from config is not present in the data. Skipping.")
                continue
                
            strategy = settings['strategy']
            
            # Check if feature has alternative encoding strategies
            if 'alternatives' in settings:
                alternatives = settings['alternatives']
                encoding_map[feature] = (strategy, alternatives)
            else:
                encoding_map[feature] = strategy
            
            # Add feature-specific parameters if they exist
            if 'params' in settings:
                encoding_params[feature] = settings['params']
                
        return encoding_map, encoding_params
        
    def _create_param_grid(self):
        """Create parameter grid for hyperparameter tuning from config.yaml file"""
        param_grid = {}
        
        # Parameters from Outlier Handling Component
        param_grid['outlier_handler__strategy'] = self.config['preprocessing']['outliers']['strategies']
        param_grid['outlier_handler__z_thresh'] = self.config['preprocessing']['outliers']['thresholds']

        # Parameters from Feature Encoding Component
        for feature, strategy_info in self.encoding_map.items():
            if isinstance(strategy_info, tuple):
                strategies = [strategy_info[0]] + strategy_info[1]
                param_grid[f'encoder__hyperparams__{feature}'] = strategies
        
        # Parameters from XGBoost Classifier Component
        param_grid['classifier__n_estimators'] = self.config['models']['xgboost']['n_estimators']
        param_grid['classifier__max_depth'] = self.config['models']['xgboost']['max_depth']
        param_grid['classifier__learning_rate'] = self.config['models']['xgboost']['learning_rate']
        param_grid['classifier__subsample'] = self.config['models']['xgboost']['subsample']
        param_grid['classifier__colsample_bytree'] = self.config['models']['xgboost']['colsample_bytree']
        param_grid['classifier__min_child_weight'] = self.config['models']['xgboost']['min_child_weight']
        param_grid['classifier__gamma'] = self.config['models']['xgboost']['gamma']
        
        # Parameters from SMOTE Component
        param_grid['smote__sampling_strategy'] = self.config['preprocessing']['target_balancing']['smote']['sampling_strategy']
        param_grid['smote__k_neighbors'] = self.config['preprocessing']['target_balancing']['smote']['k_neighbors']
        
        return param_grid
    
    
    def build_pipeline(self):
        """
        Build the imblearn pipeline with all components.
        
        Returns:
        --------
        pipeline : imblearn.pipeline.Pipeline
            The complete processing and model pipeline with proper SMOTE handling
        """
        pipeline = Pipeline([
            ('outlier_handler', OutlierHandler()),
            ('encoder', FeatureEncoder(encoding_map=self.encoding_map, encoding_params=self.encoding_params)),
            ('smote', SMOTE(random_state=self.random_state)),
            ('classifier', xgb.XGBClassifier(objective='multi:softprob', # early_stopping_rounds=10, in case of overfitting                                          
                                            random_state=self.random_state,                                           
                                            eval_metric='mlogloss',
                                            ))
        ])
        
        return pipeline
    
    def tune_hyperparameters(self, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1):
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        
        Parameters:
        -----------
        n_iter : int, default=20
            Number of hyperparameters combinations randomly sampled for evaluation (the higher the better but also the longer it will take)
            Rule of thumb: n_iter = 10* number of hyperparameters (at least)

        cv : int, default=5
            Number of cross-validation folds
            
        scoring : str, default='accuracy'
            Metric to optimize during tuning
            
        n_jobs : int, default=-1
            Number of parallel jobs (use all available cores by default)
            
        Returns:
        --------
        best_model : RandomizedSearchCV
            The fitted RandomizedSearchCV object with the best model
        """
        # Create pipeline
        pipeline = self.build_pipeline()
        
        # Create random search
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=self.param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        # Fit random search on train data - imblearn Pipeline handles SMOTE correctly
        random_search.fit(self.X_train, self.y_train)
        
        self.best_model = random_search
        
        # Print best parameters across all cross-validation folds
        print("\nBest parameters:")
        for param, value in random_search.best_params_.items():
            print(f"{param}: {value}")
        
        # Print best score (for 'scoring') across all cross-validation folds
        print(f"\nBest cross-validation score: {random_search.best_score_:.4f}")

        return random_search
    
    def evaluate_model(self):
        """
        Evaluate the best model on validation and test sets.
        
        Returns:
        --------
        dict : Dictionary with evaluation metrics
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained. Call tune_hyperparameters first.")
        
        # Make predictions on validation set
        y_val_pred = self.best_model.predict(self.X_val)
        
        # Calculate validation metrics
        val_accuracy = accuracy_score(self.y_val, y_val_pred)
        val_report = classification_report(self.y_val, y_val_pred, output_dict=True)
        
        # Make predictions on test set
        y_test_pred = self.best_model.predict(self.X_test)
        
        # Calculate test metrics
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        test_report = classification_report(self.y_test, y_test_pred, output_dict=True)
        
        # Print evaluation results
        print("\nValidation Set Metrics:")
        print(f"Accuracy: {val_accuracy:.4f}")
        print(classification_report(self.y_val, y_val_pred))
        
        print("\nTest Set Metrics:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(classification_report(self.y_test, y_test_pred))
        
        # Plot confusion matrix for test set
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(self.y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Test Set)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
        # Return metrics
        return {
            'validation': {
                'accuracy': val_accuracy,
                'report': val_report
            },
            'test': {
                'accuracy': test_accuracy,
                'report': test_report
            }
        }
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance from the best model.
        
        Parameters:
        -----------
        top_n : int, default=20
            Number of top features to display
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained. Call tune_hyperparameters first.")
        
        # Get the trained model
        model = self.best_model.best_estimator_.named_steps['classifier']
        
        # Get feature importances
        importances = model.feature_importances_
        
        try:
            # Try to get feature names directly from the model if available
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            # Alternative: try to get them from the preprocessed data
            else:
                # Apply the preprocessing steps from the pipeline to a small sample
                pipeline_without_classifier = self.best_model.best_estimator_[:-1]
                transformed_X = pipeline_without_classifier.transform(self.X_train.iloc[:1])
                
                # If it's a DataFrame with column names
                if hasattr(transformed_X, 'columns'):
                    feature_names = transformed_X.columns
                # If it's a numpy array, create generic feature names
                else:
                    feature_names = [f"Feature_{i}" for i in range(importances.shape[0])]
                    
            # Check lengths match
            if len(feature_names) != len(importances):
                print(f"Warning: Feature names length ({len(feature_names)}) doesn't match "
                      f"feature importances length ({len(importances)}). Using generic feature names.")
                feature_names = [f"Feature_{i}" for i in range(len(importances))]
                
            # Create DataFrame of feature importances
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Store feature importances
            self.feature_importances = feature_importance_df
            
            # Plot top N features
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', 
                       data=feature_importance_df.head(top_n))
            plt.title(f'Top {top_n} Feature Importances')
            plt.tight_layout()
            plt.show()
            
            return feature_importance_df
            
        except Exception as e:
            print(f"Could not plot feature importances: {str(e)}")
            print("Continuing without feature importance plot.")
            return None
    
    def run_full_pipeline(self, X, y, test_size=0.2, val_size=0.2, n_iter=20, cv=5, 
                         scoring='accuracy', custom_split=False, customer_id_col='Customer_ID'):
        """
        Run the complete pipeline from data splitting to evaluation.
        
        Parameters:
        -----------
        X : pandas DataFrame
            Input features
            
        y : pandas Series or array-like
            Target values
            
        test_size : float, default=0.2
            Proportion of data/customers to use for test set
            
        val_size : float, default=0.2
            Proportion of remaining data/customers to use for validation set
            
        n_iter : int, default=20
            Number of parameter settings sampled
            
        cv : int, default=5
            Number of cross-validation folds
            
        scoring : str, default='accuracy'
            Metric to optimize during tuning
            
        custom_split : bool, default=False
            If True, ensures all records from the same customer stay together
            If False, performs random split of individual records
            
        customer_id_col : str, default='Customer_ID'
            Name of the column containing customer IDs (only used if custom_split=True)
        """
        # Step 1: Split data
        self.split_data(X, y, test_size=test_size, val_size=val_size, 
                       custom_split=custom_split, customer_id_col=customer_id_col)
        
        # Step 2: Tune hyperparameters
        self.tune_hyperparameters(n_iter=n_iter, cv=cv, scoring=scoring)
        
        # Step 3: Evaluate model
        evaluation_metrics = self.evaluate_model()
        
        # Step 4: Plot feature importance
        self.plot_feature_importance()
        
        return evaluation_metrics