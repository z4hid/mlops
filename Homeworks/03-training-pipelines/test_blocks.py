#!/usr/bin/env python3
import os
import sys
sys.path.append('./taxi_training_pipeline')

print('Testing individual Mage blocks...')

# Test load_taxi_data
try:
    from data_loaders.load_taxi_data import load_data
    print('Loading data...')
    df = load_data()
    print(f'✓ Data loaded: {len(df):,} records')
except Exception as e:
    print(f'✗ Data loading failed: {e}')
    sys.exit(1)

# Test prepare_data  
try:
    from transformers.prepare_data import transform_data
    print('Preparing data...')
    df_prepared = transform_data(df)
    print(f'✓ Data prepared: {len(df_prepared):,} records')
except Exception as e:
    print(f'✗ Data preparation failed: {e}')
    sys.exit(1)

# Test train_model
try:
    from transformers.train_model import transform_data as train_model
    print('Training model...')
    model_info = train_model(df_prepared)
    print(f'✓ Model trained with intercept: {model_info["intercept"]:.2f}')
except Exception as e:
    print(f'✗ Model training failed: {e}')
    sys.exit(1)

# Test register_model with MLflow
try:
    from data_exporters.register_model import export_data
    print('Registering model with MLflow...')
    result = export_data(model_info)
    print(f'✓ Model registered! Size: {result["model_size"]} bytes')
except Exception as e:
    print(f'✗ Model registration failed: {e}')
    sys.exit(1)

print('✓ All blocks working! Pipeline should execute properly.') 