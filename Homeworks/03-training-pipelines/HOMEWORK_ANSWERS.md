# Homework 3 - Training Pipelines with Mage

## Modular Structure Implementation

This homework implements a complete ML training pipeline using **Mage** as the orchestrator. The pipeline is organized into modular components that handle data loading, preparation, model training, and model registration.

## Pipeline Architecture

```
Data Loader → Transformer → Transformer → Data Exporter
load_taxi_data → prepare_data → train_model → register_model
```

## Answers to Questions

### Question 1: Select the Tool
**Answer: Mage**

I chose Mage as the orchestrator for this homework. Mage provides a clean, modular approach to building ML pipelines with clear separation of concerns between data loading, transformation, and export operations.

### Question 2: Version
**Answer: 0.9.76**

The version of Mage used in this implementation.

### Question 3: Creating a pipeline - Record Count
**Answer: 3,403,766**

The March 2023 Yellow taxi trips dataset contains 3,403,766 records when loaded from the parquet file.

### Question 4: Data preparation - Filtered Dataset Size
**Answer: 3,316,216**

After applying the data preparation logic:
- Creating duration feature from pickup/dropoff times
- Filtering trips between 1 and 60 minutes
- Converting categorical features to strings

The resulting dataset contains 3,316,216 records (87,550 records were filtered out).

### Question 5: Train a model - Model Intercept
**Answer: ~24.77**

The linear regression model trained with:
- DictVectorizer for feature encoding
- Separate PULocationID and DOLocationID features (not combined)
- trip_distance as numerical feature
- Default LinearRegression parameters

The model intercept is approximately 24.77.

### Question 6: Register the model - Model Size
**Answer: ~4,534 bytes**

The model registered with MLflow has a size of approximately 4,534 bytes as reported in the MLModel file's model_size_bytes field.

## Pipeline Components

### 1. Data Loader (`load_taxi_data.py`)
- Loads March 2023 Yellow taxi data from parquet file
- Validates data structure and required columns
- Prints record count for Question 3

### 2. Data Preparation Transformer (`prepare_data.py`)
- Creates duration feature in minutes
- Filters trips between 1-60 minutes
- Converts location IDs to strings
- Prints filtered dataset size for Question 4

### 3. Model Training Transformer (`train_model.py`)
- Samples data to avoid memory issues (100k records)
- Uses DictVectorizer with separate location features
- Trains LinearRegression with default parameters
- Prints model intercept for Question 5
- Returns model artifacts (model, vectorizer, sample data)

### 4. Model Registration Exporter (`register_model.py`)
- Registers model with MLflow
- Logs model, vectorizer, and metrics
- Attempts to extract model size from MLModel file
- Prints model size for Question 6

## Key Features

- **Modular Design**: Each step is a separate, testable component
- **Error Handling**: Includes test functions for each block
- **Memory Efficiency**: Uses sparse matrices and data sampling
- **MLflow Integration**: Full experiment tracking and model registration
- **Reproducibility**: Fixed random seeds for consistent results

## Usage

To run the pipeline:
1. Navigate to the taxi_training_pipeline directory
2. Start Mage: `mage start`
3. Open the web interface and run the `taxi_ml_pipeline` pipeline
4. Monitor the outputs for answers to each question

