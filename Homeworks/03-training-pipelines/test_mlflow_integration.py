#!/usr/bin/env python3
"""
Test MLflow integration with the Mage pipeline.
This script runs the complete pipeline and verifies MLflow connectivity.
"""

import sys
import os
sys.path.append('./taxi_training_pipeline')

def test_mlflow_integration():
    print("=" * 60)
    print("Testing MLflow Integration with Mage Pipeline")
    print("=" * 60)
    
    # Test MLflow connectivity
    print("\n1. Testing MLflow Connection...")
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Test connection by listing experiments
        experiments = mlflow.search_experiments()
        print(f"✓ MLflow connected successfully")
        print(f"✓ Found {len(experiments)} experiments")
        
        # Create/get experiment
        experiment_name = "taxi-duration-prediction"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"✓ Created new experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                print(f"✓ Using existing experiment: {experiment_name}")
        except Exception as e:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✓ Created experiment: {experiment_name}")
            
    except Exception as e:
        print(f"✗ MLflow connection failed: {e}")
        return False
    
    # Run complete pipeline
    print("\n2. Running Complete Pipeline with MLflow...")
    try:
        # Load data
        from data_loaders.load_taxi_data import load_data
        print("Loading data...")
        df = load_data()
        
        # Prepare data
        from transformers.prepare_data import transform_data
        print("Preparing data...")
        df_prepared = transform_data(df)
        
        # Train model
        from transformers.train_model import transform_data as train_model
        print("Training model...")
        model_info = train_model(df_prepared)
        
        # Register with MLflow
        from data_exporters.register_model import export_data
        print("Registering model with MLflow...")
        result = export_data(model_info)
        
        print(f"✓ Pipeline completed successfully!")
        print(f"✓ Model size: {result['model_size']} bytes")
        print(f"✓ Model intercept: {result['intercept']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_mlflow_data():
    print("\n3. Verifying MLflow Data...")
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # List all experiments
        experiments = mlflow.search_experiments()
        for exp in experiments:
            print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")
            
            # Get runs for this experiment
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            print(f"  Runs: {len(runs)}")
            
            if len(runs) > 0:
                latest_run = runs.iloc[0]
                print(f"  Latest run ID: {latest_run['run_id']}")
                print(f"  Status: {latest_run['status']}")
                
                # Check if model was registered
                client = mlflow.MlflowClient()
                try:
                    model_versions = client.search_model_versions("name='taxi-duration-model'")
                    if model_versions:
                        print(f"  ✓ Model 'taxi-duration-model' registered with {len(model_versions)} version(s)")
                    else:
                        print(f"  ! No registered models found")
                except Exception as e:
                    print(f"  ! Model registration check failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ MLflow verification failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing MLflow Integration...")
    
    # Test pipeline integration
    pipeline_success = test_mlflow_integration()
    
    # Verify data was saved
    verification_success = verify_mlflow_data()
    
    if pipeline_success and verification_success:
        print("\n" + "=" * 60)
        print("✓ MLflow Integration Test: PASSED")
        print("✓ Pipeline is working correctly with MLflow!")
        print("✓ You can view results at: http://127.0.0.1:5000/")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("✗ MLflow Integration Test: FAILED")
        print("=" * 60)
        sys.exit(1) 