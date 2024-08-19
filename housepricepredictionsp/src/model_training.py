from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

def split_and_scale_data(data):
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor()
    }

    if not os.path.exists('../models'):
        os.makedirs('../models')

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f'../models/{name.replace(" ", "_").lower()}_model.pkl')

    return models
