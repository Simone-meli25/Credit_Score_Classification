"""
Data loading module for the Credit Score Classification project.
"""

import pandas as pd
import os
import numpy as np


def load_data(file_path=None):
    """
    Load the credit score dataset.
    
    Args:
        file_path (str, optional): Path to the CSV file. If None, uses default path.
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    # Use default path if none provided
    if file_path is None:
        # Get the absolute path to the data directory
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(root_dir, 'data', 'raw', 'credit_score_dataset.csv')
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    # Load data
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Display basic info
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    return df



def get_numerical_features(df):
    """
    Get numerical features from the dataframe.
    """
    return df.select_dtypes(include=['number']).columns.tolist()


def get_categorical_features(df):
    """
    Get categorical features from the dataframe.
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()














'''
def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the dataframe into training and testing sets.
    
    Args:
        df (pd.DataFrame): The dataframe to split
        target_column (str): Name of the target column
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    
    return X_train, X_test, y_train, y_test
'''