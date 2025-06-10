#!/usr/bin/env python3
"""
Simple test script to verify all pipeline blocks work correctly.
Run this to test the pipeline without using Mage UI.
"""
import sys
import os
sys.path.append('taxi_training_pipeline')

def test_pipeline():
    print("=" * 50)
    print("Testing MLOps Homework 3 Pipeline")
    print("=" * 50)
    
    try:
        # Test data loading
        print("\n1. Testing Data Loading...")
        from data_loaders.load_taxi_data import load_data
        df = load_data()
        print(f"✓ Data loaded: {len(df):,} records")
        
        # Test data preparation
        print("\n2. Testing Data Preparation...")
        from transformers.prepare_data import transform_data
        df_clean = transform_data(df)
        print(f"✓ Data cleaned: {len(df_clean):,} records remain")
        
        # Test model training
        print("\n3. Testing Model Training...")
        from transformers.train_model import transform_data as train_model
        model_info = train_model(df_clean)
        print(f"✓ Model trained with intercept: {model_info['intercept']:.2f}")
        
        # Test model registration
        print("\n4. Testing Model Registration...")
        from data_exporters.register_model import export_data
        result = export_data(model_info)
        print(f"✓ Model registered with size: {result['model_size']} bytes")
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Pipeline is working correctly.")
        print("=" * 50)
        
        # Print answers
        print("\nMLOps Homework 3 Answers:")
        print(f"Question 1: Mage")
        print(f"Question 2: 0.9.76")
        print(f"Question 3: {len(df):,}")
        print(f"Question 4: {len(df_clean):,}")
        print(f"Question 5: {model_info['intercept']:.2f}")
        print(f"Question 6: {result['model_size']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline() 