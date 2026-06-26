import numpy as np
from sklearn.metrics import mean_squared_error


# 1. RMSE, RRMSE
def rmse_scorer(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse


def rrmse_scorer(y_true, y_pred):
    rmse = rmse_scorer(y_true, y_pred)
    y_mean = np.mean(y_true)
    rrmse = (rmse / y_mean) * 100
    return rrmse


def full_scorer(estimator, X, y, scoring_function, n_regions=21):
    total_samples = len(y)
    n_years = total_samples // n_regions
    
    scores = []
    
    for i in range(n_years):
        test_index = np.arange(i * n_regions, (i + 1) * n_regions)
        train_index = np.setdiff1d(np.arange(total_samples), test_index)
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        
        score = scoring_function(y_test, y_pred)
        scores.append(score)
    
    return np.mean(scores)


# Wrapper function for BayesSearchCV
def custom_scorer(estimator, X, y, scoring_function, n_regions=21):
    return -full_scorer(estimator, X, y, scoring_function, n_regions=n_regions)
