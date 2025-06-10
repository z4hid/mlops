import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pickle
import warnings
warnings.filterwarnings('ignore')

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform_data(data, *args, **kwargs):
    """
    Train a linear regression model using DictVectorizer for feature encoding.
    """
    # Sample data for memory efficiency (100k records)
    np.random.seed(42)
    if len(data) > 100000:
        sample_data = data.sample(n=100000, random_state=42)
        print(f"Using sample of 100,000 records from {len(data):,} total records")
    else:
        sample_data = data.copy()
        print(f"Using all {len(data):,} records")
    
    # Prepare features and target
    categorical_features = ['PULocationID', 'DOLocationID']
    numerical_features = ['trip_distance']
    
    # Create feature dictionaries
    features = []
    for _, row in sample_data.iterrows():
        feature_dict = {}
        # Add categorical features
        for cat_feature in categorical_features:
            feature_dict[cat_feature] = row[cat_feature]
        # Add numerical features
        for num_feature in numerical_features:
            feature_dict[num_feature] = row[num_feature]
        features.append(feature_dict)
    
    # Target variable
    y = sample_data['duration'].values
    
    # Vectorize features
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(features)
    
    # Train linear regression model
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Get model intercept
    intercept = lr.intercept_
    print(f"Model intercept: {intercept:.6f}")
    print(f"Answer for Question 5: {intercept:.2f}")
    
    # Prepare model info for next step
    model_info = {
        'model': lr,
        'vectorizer': dv,
        'intercept': intercept,
        'n_features': X.shape[1],
        'training_samples': len(sample_data)
    }
    
    return model_info


@test
def test_output(output, *args) -> None:
    """
    Test that the model was trained successfully.
    """
    assert output is not None, 'The output is undefined'
    assert 'model' in output, 'Model not found in output'
    assert 'vectorizer' in output, 'Vectorizer not found in output'
    assert 'intercept' in output, 'Intercept not found in output'
    
    print(f"✓ Model trained successfully with intercept: {output['intercept']:.6f}") 