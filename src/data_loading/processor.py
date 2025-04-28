import pandas as pd
import os

def save_processed_data(df, filename, create_dir=True):
    """
    Save processed DataFrame to the processed data directory
    
    Args:
        df: DataFrame to save
        filename: Name of the file (without path)
        create_dir: Create the directory if it doesn't exist
    """
    # Get project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(root_dir, 'data', 'processed')
    
    # Create directory if it doesn't exist
    if create_dir and not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    # Save the data
    file_path = os.path.join(processed_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Saved processed data to {file_path}")
    
    return file_path