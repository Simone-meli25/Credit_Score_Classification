from src.utils.config_loader import load_config

def create_encoding_map_from_config(config):
    """Create encoding map from config dictionary"""
    encoding_map = {}
    encoding_params = {}
    
    for feature, feature_config in config['encoding']['features'].items():
        strategy = feature_config['strategy']
        
        # Check if this feature has alternatives
        if 'alternatives' in feature_config:
            encoding_map[feature] = (strategy, feature_config['alternatives'])
        else:
            encoding_map[feature] = strategy
        
        # Store any params
        if 'params' in feature_config:
            if feature not in encoding_params:
                encoding_params[feature] = {}
            encoding_params[feature].update(feature_config['params'])
    
    return encoding_map, encoding_params

def create_param_grid_from_config(config):
    """Create parameter grid for RandomizedSearchCV from config"""
    param_grid = {}

        # Update in create_param_grid_from_config function
    if 'preprocessing' in config:
        if 'outliers' in config['preprocessing']:
            outlier_params = config['preprocessing']['outliers']
            param_grid['outlier_handler__strategy'] = outlier_params['strategies']
            param_grid['outlier_handler__z_thresh'] = outlier_params['thresholds']

        if 'encoding' in config['preprocessing']:
            #TODO
            pass

    # And for model parameters
    if 'models' in config:
        for model_name, model_params in config['models'].items():
            if model_name == 'random_forest':
                for param_name, param_values in model_params.items():
                    param_grid[f'classifier__{param_name}'] = param_values
    
    return param_grid

# Main code
def build_pipeline():
    # Load configuration
    config = load_config()
    
    # Create encoding map and params from config
    encoding_map, encoding_params = create_encoding_map_from_config(config)
    
    # Create parameter grid for hyperparameter tuning
    param_grid = create_param_grid_from_config(config)
    
    # Create pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from outlier_handler import OutlierHandler
    from feature_encoder import FeatureEncoder
    
    pipeline = Pipeline([
        ('outlier_handler', OutlierHandler()),
        ('encoder', FeatureEncoder(encoding_map, encoding_params)),
        ('classifier', RandomForestClassifier())
    ])
    
    return pipeline, param_grid

# Use in randomized search
def run_search(X, y):
    from sklearn.model_selection import RandomizedSearchCV
    
    pipeline, param_grid = build_pipeline()
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        random_state=42
    )
    
    random_search.fit(X, y)
    return random_search







# Step 5: Use it in your main script

import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('your_data.csv')

# Split into features and target
X = df.drop('target_column', axis=1)
y = df['target_column']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run the search
from your_module import run_search

search_results = run_search(X_train, y_train)

# Get best model
best_model = search_results.best_estimator_

# Evaluate on test set
score = best_model.score(X_test, y_test)
print(f"Best model test accuracy: {score:.4f}")
print(f"Best parameters: {search_results.best_params_}")