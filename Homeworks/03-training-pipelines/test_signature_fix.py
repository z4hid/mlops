#!/usr/bin/env python3
"""
Test script to verify that MLflow signature warning is fixed.
This script runs just the model registration part to check for warnings.
"""

import sys
import os
import warnings
sys.path.append('./taxi_training_pipeline')

def test_signature_fix():
    print("🔍 Testing MLflow Signature Fix...")
    print("=" * 50)
    
    # Load some sample data quickly
    import pandas as pd
    sample_data = pd.DataFrame([
        {'PULocationID': '161', 'DOLocationID': '236', 'trip_distance': 2.5, 'duration': 15.2},
        {'PULocationID': '43', 'DOLocationID': '151', 'trip_distance': 1.8, 'duration': 8.1},
        {'PULocationID': '79', 'DOLocationID': '145', 'trip_distance': 3.2, 'duration': 18.5}
    ])
    
    print(f"✅ Created sample data: {len(sample_data)} records")
    
    # Train a simple model
    try:
        from transformers.train_model import transform_data as train_model
        print("🤖 Training model...")
        model_info = train_model(sample_data)
        print(f"✅ Model trained with intercept: {model_info['intercept']:.2f}")
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False
    
    # Test model registration with signature
    try:
        print("📝 Registering model with MLflow...")
        
        # Capture warnings to check if signature warning appears
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            from data_exporters.register_model import export_data
            result = export_data(model_info)
            
            # Check for signature-related warnings
            signature_warnings = [warning for warning in w 
                                if "signature" in str(warning.message).lower() 
                                or "input_example" in str(warning.message).lower()]
            
            if signature_warnings:
                print("⚠️  Found signature warnings:")
                for warning in signature_warnings:
                    print(f"    {warning.message}")
                return False
            else:
                print("✅ No signature warnings found!")
                print(f"✅ Model registered successfully (size: {result['model_size']} bytes)")
                return True
                
    except Exception as e:
        print(f"❌ Model registration failed: {e}")
        return False

def main():
    print("🧪 MLflow Signature Fix Verification")
    print("=" * 50)
    
    success = test_signature_fix()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 SUCCESS: MLflow signature warning has been FIXED!")
        print("✅ Models now log with proper signatures and input examples")
        print("✅ No more 'Model logged without a signature' warnings")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ FAILED: Signature warning still present")
        print("=" * 50)
        sys.exit(1)

if __name__ == "__main__":
    main() 