import pandas as pd
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs):
    """
    Load March 2023 Yellow taxi trips data from parquet file.
    """
    # Load the data - corrected path to find the data file
    file_path = '../data/yellow_tripdata_2023-03.parquet'
    
    # Try alternative paths if the default doesn't work
    import os
    if not os.path.exists(file_path):
        # Try from current directory when running tests
        file_path = 'data/yellow_tripdata_2023-03.parquet'
        if not os.path.exists(file_path):
            # Try absolute path if relative paths fail
            file_path = '/home/z4hid/Desktop/githubProjects/mlops/Homeworks/03-training-pipelines/data/yellow_tripdata_2023-03.parquet'
    
    print(f"Loading data from: {file_path}")
    df = pd.read_parquet(file_path)
    
    # Print the number of records loaded (Question 3)
    num_records = len(df)
    print(f"Number of records loaded: {num_records:,}")
    print(f"Answer for Question 3: {num_records}")
    
    return df


@test
def test_output(output, *args) -> None:
    """
    Test that the output is not empty and has expected columns.
    """
    assert output is not None, 'The output is undefined'
    assert len(output) > 0, 'The output is empty'
    
    # Check for required columns
    required_columns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'PULocationID', 'DOLocationID']
    for col in required_columns:
        assert col in output.columns, f'Required column {col} not found'
    
    print(f"✓ Data loaded successfully with {len(output):,} records") 