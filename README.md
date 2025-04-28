# Credit Score Classification

A supervised machine learning pipeline for credit score analysis that uses advanced preprocessing techniques and machine learning models to classify credit scores.

## Project Structure

- **src/**: Core implementation modules
  - **data_loading/**: Data loading and initial processing
    - `data_loader.py`: Functions for loading datasets
  - **preprocessing/**: Data cleaning and feature engineering
    - **missing_values/**: Handling missing data
      - `missing_values_processing.py`: General missing value handling
      - `knn_imputation.py`: KNN-based imputation techniques (structure in place - to complete)
    - **categorical/**: Categorical feature processing
      - `categorical_features.py`: cleaning categorical variables
      - `problematic_numeric_values.py`: Handling categorical features with numeric values to clean
      - `features_encoding.py`: categorical features encoding (TODO)
    - **numeric_features/**: Numeric feature processing
      - `numeric_features_processing.py`: handling inconsistencies in the numeric values
    - **outliers/**: Outlier detection and handling (TODO)
  - **models/**: Machine learning models
    - `credit_score_classifier.py`: Model implementation (TODO)
  - **utils/**: Utility functions
    - `helper.py`: General utility functions (structure in place - if needed)
    - `visualization.py`: Data visualization utilities (structure in place - if needed)
  - `main.py`: Main pipeline orchestration (structure in place - if needed)
- **data/**: Raw and processed datasets
  - **raw/**: Original data
  - **processed/**: Cleaned and transformed data (for adding here the dataset after the preprocessing)
- **notebooks/**: Exploratory analysis and visualization
  - `new_version.ipynb`: Current notebook to work on
  - `old_version.ipynb`: Previous notebook
- **tests/**: Unit tests (in case of need)
  - **preprocessing/**: Tests for preprocessing modules
  - **models/**: Tests for machine learning models
  - **utils/**: Tests for utility functions

## Current Status

### Completed
- Basic project structure and architecture
- Data loading functionality
- Initial exploratory data analysis
- Categorical feature processing 
- Numeric feature processing for inconsistencies


### In Progress / To Do
- Missing value handling with advanced imputation techniques (mainly KNN)
- Outliers Handling
- Correlation Analysis and eventual Features selection
- Categorical Features encoding
- Features Scaling (optional depending on the chosen model to train)
- Models training
- Models evaluation
