import mlflow
import pickle
import tempfile
import os
import pandas as pd
import numpy as np
from datetime import datetime
from mlflow.models.signature import infer_signature

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Register the trained model with MLflow and calculate its size.
    """
    model_info = data
    
    # Set MLflow tracking URI to local SQLite database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Set experiment
    experiment_name = "taxi-duration-prediction"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("vectorizer_type", "DictVectorizer")
        mlflow.log_param("n_features", model_info['n_features'])
        mlflow.log_param("training_samples", model_info['training_samples'])
        
        # Log metrics
        mlflow.log_metric("intercept", model_info['intercept'])
        
        # Create input example and signature
        # Create sample input data that matches the training format
        input_example = pd.DataFrame([
            {
                'PULocationID': '161',
                'DOLocationID': '236', 
                'trip_distance': 2.5
            },
            {
                'PULocationID': '43',
                'DOLocationID': '151',
                'trip_distance': 1.8
            }
        ])
        
        # Transform the input example using the vectorizer to match model expectations
        vectorizer = model_info['vectorizer']
        model = model_info['model']
        
        # Convert to dictionary format as expected by the vectorizer
        input_dicts = input_example.to_dict('records')
        X_example = vectorizer.transform(input_dicts)
        
        # Generate predictions for signature inference
        example_predictions = model.predict(X_example)
        
        # Infer signature from input and output
        signature = infer_signature(X_example.toarray(), example_predictions)
        
        # Create a temporary file to save the model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            # Save both model and vectorizer together
            model_data = {
                'model': model_info['model'],
                'vectorizer': model_info['vectorizer']
            }
            pickle.dump(model_data, tmp_file)
            tmp_file.flush()
            
            # Get file size
            model_size = os.path.getsize(tmp_file.name)
            
            # Log the model with input example and signature
            mlflow.sklearn.log_model(
                sk_model=model_info['model'],
                artifact_path="model",
                registered_model_name="taxi-duration-model",
                input_example=X_example.toarray(),
                signature=signature
            )
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
    
    print(f"Model registered successfully!")
    print(f"Model size: {model_size} bytes")
    print(f"Answer for Question 6: {model_size}")
    
    return {
        'model_size': model_size,
        'intercept': model_info['intercept'],
        'experiment_id': experiment_id
    } 