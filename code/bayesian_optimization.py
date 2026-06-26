import os
import yaml
import pandas as pd

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from scorer import custom_scorer


# Load all YAML files from a directory
def load_param_spaces_from_dir(directory):
    param_spaces = {}
    for filename in os.listdir(directory):
        if filename.endswith('.yaml'):
            model_name = os.path.splitext(filename)[0]
            with open(os.path.join(directory, filename), 'r') as file:
                param_spaces[model_name] = yaml.safe_load(file)
    return param_spaces

def convert_to_skopt_space(params_dict):
    skopt_space = {}
    search_space = params_dict.get('search_space', {})

    if isinstance(search_space, list):  # 리스트로 정의된 경우
        for param_set in search_space:
            for key, value in param_set.items():
                if value['type'] == 'Real':
                    skopt_space[key] = Real(value['low'], value['high'], prior=value.get('prior', 'uniform'))
                elif value['type'] == 'Integer':
                    skopt_space[key] = Integer(value['low'], value['high'])
                elif value['type'] == 'Categorical':
                    skopt_space[key] = Categorical(value['categories'])
                else:
                    print(f"Warning: Unrecognized type {value['type']} for parameter {key}")
    elif isinstance(search_space, dict):  # 딕셔너리로 정의된 경우
        for key, value in search_space.items():
            if value['type'] == 'Real':
                skopt_space[key] = Real(value['low'], value['high'], prior=value.get('prior', 'uniform'))
            elif value['type'] == 'Integer':
                skopt_space[key] = Integer(value['low'], value['high'])
            elif value['type'] == 'Categorical':
                skopt_space[key] = Categorical(value['categories'])
            else:
                print(f"Warning: Unrecognized type {value['type']} for parameter {key}")
    else:
        print("Warning: search_space is neither a list nor a dictionary")

    return skopt_space

# Mapping of model names to their corresponding classes
model_mapping = {
    'svm': SVR,
    'rf': RandomForestRegressor,
    'gb': GradientBoostingRegressor,
    'xgb': XGBRegressor,
    'dt': DecisionTreeRegressor,
    'knn': KNeighborsRegressor
}

def clean_and_convert_params(params):
    cleaned_params = {}
    for key, value in params.items():
        if isinstance(value, float) and value.is_integer():
            cleaned_params[key] = int(value)
        elif value != value:  # Checks for NaN values
            cleaned_params[key] = None
        else:
            cleaned_params[key] = value
    return cleaned_params

# Perform BayesSearchCV for each model
def run_bayes_search(X, y, param_spaces, output_dir, scoring_function, n_regions=21):
    os.makedirs(output_dir, exist_ok=True)

    for model_name, search_space in param_spaces.items():
        print(f"Running BayesSearchCV for {model_name.upper()}...")
        skopt_space = convert_to_skopt_space(search_space)
        
        print(f"Search space for {model_name.upper()}: {skopt_space}")
        
        if not skopt_space:
            print(f"Warning: Search space for {model_name.upper()} is empty. Skipping...")
            continue
        
        model_class = model_mapping[model_name]
        
        bayes_search = BayesSearchCV(
            estimator=model_class(),
            search_spaces=skopt_space,
            scoring=lambda estimator, X, y: custom_scorer(estimator, X, y, scoring_function, n_regions=n_regions),
            n_jobs=-1,
            verbose=1,
            n_iter=50,
            random_state=42
        )
        
        bayes_search.fit(X, y)
        
        # Clean and convert params to appropriate types
        best_params = clean_and_convert_params(bayes_search.best_params_)
        
        output_file = os.path.join(output_dir, f'{model_name}_best_params.csv')
        pd.DataFrame([best_params]).to_csv(output_file, index=False)
        print(f"Best parameters for {model_name.upper()} saved to {output_file}")
