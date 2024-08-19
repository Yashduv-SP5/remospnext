from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib

def optimize_model(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    model_rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    joblib.dump(grid_search.best_estimator_, '../models/best_random_forest_model.pkl')

    return grid_search.best_params_
