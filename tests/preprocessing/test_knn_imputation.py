import pandas as pd
import numpy as np
from src.preprocessing.missing_values.knn_imputation import (
    impute_knn,
    create_holdout_mask,
    apply_and_evaluate_knn_imputation,
    apply_and_evaluate_knn_for_all_numeric_columns
)

import pytest

@pytest.fixture
def test_df():
    """Create a test dataframe with missing values."""
    np.random.seed(42)
    return pd.DataFrame({
        'numeric1': [1, 2, np.nan, 4, 5, 6, np.nan, 8, None, 10],
        'numeric2': [10, np.nan, 30, 40, 50, np.nan, 70, 80, 90, 100],
        'categorical': ['a', 'b', 'c', None, 'e', 'f', 'g', np.nan, 'i', 'j'],
        'numeric3': [100, 200, None, 400, np.nan, 600, 700, 800, 900, 1000],
    })

@pytest.fixture
def missing_cols():
    """List of columns with missing values."""
    return ['numeric1', 'numeric2', 'categorical', 'numeric3']

def test_impute_knn(test_df, missing_cols):
    """Test knn imputation for a single column."""

    for col in missing_cols:
        
        # Test with standard parameters
        result_df = impute_knn(test_df, column_to_impute=col, n_neighbors=3)
        
        # Check that the result is a dataframe
        assert isinstance(result_df, pd.DataFrame)

        # Check that missing values are filled
        assert result_df[col].isna().sum() == 0

         # Check that non-target columns aren't modified
        for other_col in test_df.columns:
            if other_col != col:  # Skip the column being imputed
                pd.testing.assert_series_equal(test_df[other_col], result_df[other_col])


        # Test with different neighbors parameter
        result_df2 = impute_knn(test_df, column_to_impute=col, n_neighbors=1)
        # Values should be different with different k values
        assert not result_df[col].equals(result_df2[col])


def test_create_holdout_mask(test_df, missing_cols):
    """Test creation of holdout mask."""

    for col in missing_cols:
        # Test with default holdout fraction
        mask, values = create_holdout_mask(test_df, col)
        
        # Check that mask is boolean array
        assert mask.dtype == bool
        
        # Check that mask length matches dataframe
        assert len(mask) == len(test_df)
        
        # Check that holdout values are from non-missing values
        assert values.isna().sum() == 0
        
        # Check that approximately 20% of non-missing values are held out
        non_missing_count = (~test_df[col].isna()).sum()
        expected_holdout = int(non_missing_count * 0.2)
        assert len(values) == expected_holdout
        
        # Test with different holdout fraction
        mask2, values2 = create_holdout_mask(test_df, col, holdout_fraction=0.5)
        expected_holdout2 = int(non_missing_count * 0.5)
        assert len(values2) == expected_holdout2

def test_apply_and_evaluate_knn_imputation(test_df, missing_cols):
    """Test evaluation of KNN imputation for a specific column."""

    for col in missing_cols:
        # Test with default parameters
        imputed_df, results = apply_and_evaluate_knn_imputation(test_df.copy(), col)
        
        # Check that imputed dataframe is returned
        assert isinstance(imputed_df, pd.DataFrame)
        
        # Check that results dictionary has expected metrics
        expected_metrics = ['rmse', 'delta_mean', 'delta_std', 'delta_25', 
                            'delta_75', 'distribution_score', 'combined_score']
        for metric in expected_metrics:
            assert metric in results
        
        # Test with different parameters
        imputed_df2, results2 = apply_and_evaluate_knn_imputation(
            test_df.copy(), col, holdout_fraction=0.3, k=5)
        
        # Results should be different with different parameters
        assert results['rmse'] != results2['rmse']

def test_apply_and_evaluate_knn_for_all_numeric_columns(test_df, missing_cols):
    """Test KNN imputation evaluation for all numeric columns."""
    # Test with default parameters
    result_df = apply_and_evaluate_knn_for_all_numeric_columns(
        test_df, missing_cols)
    
    # Check that result is a dataframe
    assert isinstance(result_df, pd.DataFrame)
    
    # Check that it processed all missing columns
    for col in missing_cols:
        # Missing values should be reduced or eliminated in the processed columns
        assert result_df[col].isna().sum() <= test_df[col].isna().sum()

def test_edge_cases(test_df):
    """Test edge cases for KNN imputation."""
    # Test with empty dataframe
    empty_df = pd.DataFrame()
    with pytest.raises(Exception):
        impute_knn(empty_df, 'column')
    
    # Test with no missing values
    no_missing_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = impute_knn(no_missing_df, 'A')
    pd.testing.assert_frame_equal(result, no_missing_df)
    
    # Test with column not in dataframe
    with pytest.raises(Exception):
        impute_knn(test_df, 'nonexistent_column')