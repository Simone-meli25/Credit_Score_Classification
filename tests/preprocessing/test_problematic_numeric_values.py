"""
Tests for problematic numeric values.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np


# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.preprocessing.categorical.problematic_numeric_values import (
    convert_non_numeric_strings_to_nan,
    identify_problematic_characters,
    remove_characters
)



def test_convert_non_numeric_strings_to_nan():
    """Test that convert_non_numeric_strings_to_nan correctly converts strings."""
    # Create a test DataFrame
    df = pd.DataFrame({
        'A': ['abc', 'def123', 'g_hi', '456', 'jkl789'],
        'B': ['xyz', '123abc', 'uvw', '789', 'rs$t']
    })
    
    # Run the function
    result = convert_non_numeric_strings_to_nan(df, ['A', 'B'])
    
    # Verify strings without digits are converted to NaN
    assert pd.isna(result['A'][0])  # 'abc' -> NaN
    assert result['A'][1] == 'def123'  # Contains digits, unchanged
    assert pd.isna(result['A'][2])  # 'g_hi' -> NaN
    assert result['A'][3] == '456'  # Only digits, unchanged
    assert result['A'][4] == 'jkl789'  # Contains digits, unchanged
    
    assert pd.isna(result['B'][0])  # 'xyz' -> NaN
    assert result['B'][1] == '123abc'  # Contains digits, unchanged
    assert pd.isna(result['B'][2])  # 'uvw' -> NaN
    assert result['B'][3] == '789'  # Only digits, unchanged
    assert pd.isna(result['B'][4])  # 'rs$t' -> NaN


def test_identify_problematic_characters():
    """Test that identify_problematic_characters correctly identifies problematic characters."""
    # Create a test DataFrame with various problematic characters
    df = pd.DataFrame({
        'A': [123, '456$', '-789_', np.nan, '10.5', '2030'],
        'B': ['100%', 'dog', '300*', '400+', np.nan, '60,7']
    })
    
    # Run the function
    problematic = identify_problematic_characters(df, ['A', 'B'])
    
    # Verify problematic characters are identified
    assert '$' in problematic
    assert '_' in problematic
    assert '-' not in problematic
    assert '%' in problematic
    assert '*' in problematic
    assert '+' in problematic
    
    # Characters should be identified as problematic
    assert 'd' in problematic  
    assert 'o' in problematic
    assert 'g' in problematic

    assert np.nan not in problematic
    assert 'n' not in problematic
    assert 'a' not in problematic



def test_remove_characters():
    """Test that remove_characters correctly removes characters from values."""
    # Create a test DataFrame
    df = pd.DataFrame({
        'A': ['100_', '20$0_', np.nan, '__400_'],
        'B': ['500_', '_ 600', '700_', 800]
    })
    
    # Run the function
    result = remove_characters(df, ['A', 'B'], characters_to_remove=['_', '$'])
    
    # Verify characters were removed
    assert result['A'].tolist() == ['100', '200', np.nan, '400']
    assert result['B'].tolist() == ['500', ' 600', '700', '800']
    
    # Ensure the original DataFrame is unchanged
    assert df['A'].tolist() == ['100_', '20$0_', np.nan, '__400_']
    assert df['B'].tolist() == ['500_', '_ 600', '700_', 800]





