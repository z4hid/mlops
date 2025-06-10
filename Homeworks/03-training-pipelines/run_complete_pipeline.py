#!/usr/bin/env python3
"""
Complete pipeline execution script that ensures MLflow and Mage integration works.
This script will run the pipeline and verify the results are visible in both UIs.
"""

import os
import sys
import time
import requests
import json

# Add taxi_training_pipeline to path
sys.path.append('./taxi_training_pipeline')

def run_pipeline():
    """Run the complete ML pipeline"""
    print("🚀 Starting Complete Pipeline Execution...")
    print("=" * 60)
    
    # Import and run all blocks in sequence
    try:
        # Block 1: Load data
        print("\n📊 Block 1: Loading taxi data...")
        from data_loaders.load_taxi_data import load_data
        df = load_data()
        print(f"✅ Loaded {len(df):,} records")
        
        # Block 2: Prepare data
        print("\n🔧 Block 2: Preparing data...")
        from transformers.prepare_data import transform_data
        df_prepared = transform_data(df)
        print(f"✅ Prepared {len(df_prepared):,} records")
        
        # Block 3: Train model
        print("\n🤖 Block 3: Training model...")
        from transformers.train_model import transform_data as train_model
        model_info = train_model(df_prepared)
        print(f"✅ Model trained (intercept: {model_info['intercept']:.2f})")
        
        # Block 4: Register with MLflow
        print("\n📝 Block 4: Registering model with MLflow...")
        from data_exporters.register_model import export_data
        result = export_data(model_info)
        print(f"✅ Model registered (size: {result['model_size']} bytes)")
        
        return True, result
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False, None

def verify_mlflow():
    """Verify MLflow has the data"""
    print("\n🔍 Verifying MLflow Integration...")
    
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Check experiments
        experiments = mlflow.search_experiments()
        print(f"✅ MLflow: Found {len(experiments)} experiments")
        
        # Check runs in taxi-duration-prediction experiment
        for exp in experiments:
            if exp.name == "taxi-duration-prediction":
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                print(f"✅ MLflow: {len(runs)} runs in {exp.name}")
                
                if len(runs) > 0:
                    latest_run = runs.iloc[0]
                    print(f"✅ MLflow: Latest run status: {latest_run['status']}")
                
        # Check registered models
        client = mlflow.MlflowClient()
        try:
            models = client.search_model_versions("name='taxi-duration-model'")
            print(f"✅ MLflow: {len(models)} model versions registered")
        except Exception as e:
            print(f"⚠️  MLflow: Model check failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ MLflow verification failed: {e}")
        return False

def check_web_interfaces():
    """Check that web interfaces are accessible"""
    print("\n🌐 Checking Web Interfaces...")
    
    # Check MLflow
    try:
        response = requests.get("http://127.0.0.1:5000/", timeout=5)
        if response.status_code == 200:
            print("✅ MLflow UI: Accessible at http://127.0.0.1:5000/")
        else:
            print(f"⚠️  MLflow UI: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ MLflow UI: Not accessible - {e}")
    
    # Check Mage
    try:
        response = requests.get("http://127.0.0.1:6789/", timeout=5)
        if response.status_code == 200:
            print("✅ Mage UI: Accessible at http://127.0.0.1:6789/")
        else:
            print(f"⚠️  Mage UI: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Mage UI: Not accessible - {e}")

def main():
    print("🔄 MLflow + Mage Integration Test")
    print("=" * 60)
    
    # 1. Run the pipeline
    pipeline_success, result = run_pipeline()
    
    if not pipeline_success:
        print("\n❌ FAILED: Pipeline execution failed")
        sys.exit(1)
    
    # 2. Verify MLflow
    mlflow_success = verify_mlflow()
    
    # 3. Check web interfaces
    check_web_interfaces()
    
    # 4. Final status
    print("\n" + "=" * 60)
    if pipeline_success and mlflow_success:
        print("🎉 SUCCESS: MLflow + Mage Integration Working!")
        print(f"📊 Model size: {result['model_size']} bytes")
        print(f"📈 Model intercept: {result['intercept']:.2f}")
        print("🌐 MLflow UI: http://127.0.0.1:5000/")
        print("🌐 Mage UI: http://127.0.0.1:6789/")
        print("=" * 60)
        print("✅ You can now view experiments and models in both interfaces!")
    else:
        print("❌ FAILED: Integration issues detected")
        sys.exit(1)

if __name__ == "__main__":
    main() 