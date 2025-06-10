import pandas as pd
if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform_data(data, *args, **kwargs):
    """
    Prepare the data for training by calculating duration and filtering.
    """
    df = data.copy()
    
    # Calculate duration in minutes
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Filter records with duration between 1 and 60 minutes
    initial_count = len(df)
    df_filtered = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()
    final_count = len(df_filtered)
    
    # Convert categorical columns to string
    categorical_columns = ['PULocationID', 'DOLocationID']
    for col in categorical_columns:
        df_filtered[col] = df_filtered[col].astype(str)
    
    print(f"Initial records: {initial_count:,}")
    print(f"Records after filtering: {final_count:,}")
    print(f"Records removed: {initial_count - final_count:,}")
    print(f"Answer for Question 4: {final_count}")
    
    return df_filtered


@test
def test_output(output, *args) -> None:
    """
    Test that the output has duration column and no invalid durations.
    """
    assert output is not None, 'The output is undefined'
    assert 'duration' in output.columns, 'Duration column not found'
    assert output['duration'].min() >= 1, 'Duration less than 1 minute found'
    assert output['duration'].max() <= 60, 'Duration greater than 60 minutes found'
    
    print(f"✓ Data prepared successfully with {len(output):,} records") 