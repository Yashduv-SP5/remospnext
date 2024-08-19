import pandas as pd

def feature_engineering(data):
    # Apply One-Hot Encoding
    categorical_columns = data.select_dtypes(include=['object']).columns
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Create a feature for house age
    data_encoded['HouseAge'] = data_encoded['YearBuilt'] - data_encoded['YearRemodAdd']

    return data_encoded
