# Credit_Score_Classification
Supervised Machine Learning pipeline for Credit Score Analysis

## Project Structure
- `src/`: Core implementation modules
  - `preprocessing/`: Data cleaning and feature engineering
  - `models/`: Machine learning models
  - `utils/`: Utility functions
- `data/`: Raw and processed datasets
- `notebooks/`: Exploratory analysis and visualization
- `tests/`: Unit tests

## Testing
This project uses pytest for unit testing:

```
# Install requirements
pip install -r requirement.txt

# Run tests with coverage report
.\run_tests.bat

# Run tests manually
python -m pytest tests -v
```

The test suite includes tests for:
- Missing value handling
- Categorical feature processing
- Data validation
- Main pipeline functions

