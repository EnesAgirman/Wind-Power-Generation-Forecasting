import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
def preprocess_dataset():
    """
    Preprocess the dataset and save the train, validation and test datasets.
    :return: train_dataset, validation_dataset, test_dataset
    """
    warnings.filterwarnings("ignore")
    
    # Initialize the work and data directories
    work_dir = os.getcwd()
    data_dir = 'Data'
    
    # Read the features and power data from the csv files in the data directory and save them as pandas dataframes
    features = pd.read_csv(f'{data_dir}/features.csv', parse_dates=['Timestamp'], index_col=['Timestamp'])
    power = pd.read_csv(f'{data_dir}/power.csv', parse_dates=['Timestamp'], index_col=['Timestamp'])

    # Merge the features and power datasets and drop the Power(kW) column
    dataset = pd.merge(features, power, on=['Timestamp'])
    columns_before = dataset.columns.drop(['Power(kW)'])
    
    # Fill the missing values with the previous values
    dataset.fillna(method='ffill', inplace=True)

    # Standard and MinMaxScaler for all datasets
    standard_scaler = StandardScaler()
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    # standardize the features in the dataset using the StandardScaler
    dataset = pd.DataFrame(standard_scaler.fit_transform(dataset), columns=dataset.columns, index=dataset.index)

    # Get the date and time features from the dataset and get the sin and cos of these features
    dataset["sin_hour_of_day"] = np.sin(2 * np.pi * dataset.index.hour / 24)
    dataset["cos_hour_of_day"] = np.cos(2 * np.pi * dataset.index.hour / 24)
    dataset["sin_month_of_year"] = np.sin(2 * np.pi * dataset.index.month / 12)
    dataset["cos_month_of_year"] = np.cos(2 * np.pi * dataset.index.month / 12)
    dataset["sin_day_of_year"] = np.sin(2 * np.pi * dataset.index.dayofyear / 365)
    dataset["cos_day_of_year"] = np.cos(2 * np.pi * dataset.index.dayofyear / 365)
    
    # Initialize the y variable as the power column from the dataset. 
    # Then we drop the power column from the dataset and add it back to make it the last column on the dataset
    y = dataset["Power(kW)"]
    dataset.drop(['Power(kW)'], axis=1, inplace=True)
    dataset["Power(kW)"] = y
    columns_after_date_features = dataset.columns

    # Add the lagged features to the dataset
    for i in range(1, 7):
        for column in columns_after_date_features:
            dataset[f"lag_{i}_{column}"] = dataset[column].shift(i)

    lagged_columns = [column for column in dataset.columns if "lag" in column]
    
    # Add the rolling mean and rolling std features to the dataset
    for column in lagged_columns:
        dataset[f"rolling_mean_{6}_{column}"] = dataset[column].rolling(6).mean()
        dataset[f"rolling_std_{6}_{column}"] = dataset[column].rolling(6).std()

    # Drop the rows with missing values
    dataset.dropna(inplace=True)
    
    # Drop the power column from the dataset and also drop the columns before the lagged features
    dataset.drop(['Power(kW)'], axis=1, inplace=True)
    dataset.drop(columns_before, axis=1, inplace=True)

    # Add the power column back to the dataset to make it the last column and drop the rows with missing values
    dataset["Power(kW)"] = y
    dataset.dropna(inplace=True)
    
    # Train, Validation, Test Split
    # Split the dataset into train, validation and test datasets
    train_dataset, validation_dataset = train_test_split(dataset, test_size=0.2, shuffle=False)
    validation_dataset, test_dataset = train_test_split(validation_dataset, test_size=0.5, shuffle=False)
    
    # Save the datasets
    # Save the preprocessed train, validation and test datasets as csv files in the data directory
    train_dataset = pd.DataFrame(train_dataset, columns=dataset.columns, index=dataset.index[:len(train_dataset)])
    validation_dataset = pd.DataFrame(validation_dataset, columns=dataset.columns,
                                      index=dataset.index[len(train_dataset):len(train_dataset) + len(validation_dataset)])
    test_dataset = pd.DataFrame(test_dataset, columns=dataset.columns,
                                index=dataset.index[len(train_dataset) + len(validation_dataset):])

    train_dataset.to_csv(f'{data_dir}/train_dataset.csv')
    validation_dataset.to_csv(f'{data_dir}/validation_dataset.csv')
    test_dataset.to_csv(f'{data_dir}/test_dataset.csv')

    # Return the preprocessed train, validation and test datasets
    return train_dataset, validation_dataset, test_dataset