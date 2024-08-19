import pandas as pd

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Drop rows where 'SalePrice' is missing
    data = data.dropna(subset=['SalePrice'])

    # Fill missing values for categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        data[column] = data[column].fillna(data[column].mode()[0])

    # Fill missing values for numerical columns
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    for column in numerical_columns:
        data[column] = data[column].fillna(data[column].mean())

    return data
