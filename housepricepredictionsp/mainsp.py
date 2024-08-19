from src.data_preparation import load_and_preprocess_data
from src.feature_engineering import feature_engineering
from src.model_training import split_and_scale_data, train_models
from src.model_optimization import optimize_model
from src.utils import evaluate_model
import joblib

def main():
    # Load and preprocess data
    data = load_and_preprocess_data('data/house_prices.csv')

    # Feature engineering
    data_encoded = feature_engineering(data)

    # Split and scale data
    X_train, X_test, y_train, y_test = split_and_scale_data(data_encoded)

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate models
    for name, model in models.items():
        print(f"Evaluating {name}")
        evaluate_model(model, X_test, y_test)

    # Optimize model
    best_params = optimize_model(X_train, y_train)
    print(f"Best parameters for Random Forest: {best_params}")

if __name__ == "__main__":
    main()
