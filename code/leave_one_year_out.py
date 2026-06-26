import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import load_best_params, plot_avg_metric_by_region, plot_results, plot_yield_estimation


years = np.arange(2010, 2023 + 1)

# 2. leave_one_year_out
# y_pred와 y_test를 반환
def leave_one_year_out(X, y, estimator, params, region_names, selected_years=None):
    results_list = []
    n_regions = len(region_names)

    if selected_years is None:
        selected_years = years  # 모든 연도 선택
    
    for i, test_year in enumerate(selected_years):
        # 테스트 세트와 학습 세트 분리
        test_index = np.arange(i * n_regions, (i + 1) * n_regions)
        train_index = np.setdiff1d(np.arange(len(y)), test_index)
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 모델 생성 및 학습
        regressor = estimator(**params)
        regressor.fit(X_train, y_train)
        
        # 테스트 세트에 대한 예측
        y_pred = regressor.predict(X_test)
        
        # 지역별 결과를 리스트에 추가
        for j in range(len(test_index)):
            region_name = region_names[j % n_regions]
            y_test_region = y_test[j]
            y_pred_region = y_pred[j]
            
            results_list.append({
                'region_name': region_name,
                'year': test_year,
                'y_test': float(y_test_region),
                'y_pred': float(y_pred_region)
            })
    
    results_df = pd.DataFrame(results_list)
    return results_df

def calculate_metrics_from_df(results_df, output_csv_path='metric.csv'):
    # Initialize a list to store the metrics for each year
    metrics_list = []

    # Get the unique years from the dataframe
    years = results_df['year'].unique()

    # Calculate metrics for each year
    for year in years:
        # Filter the dataframe for the current year
        year_df = results_df[results_df['year'] == year]
        
        # Extract the actual and predicted values for the year
        y_test = year_df['y_test'].values
        y_pred = year_df['y_pred'].values

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        rrmse = (rmse / np.mean(y_test)) * 100

        # Append metrics and the year to the list
        metrics_list.append({
            'year': year,
            'RRMSE': rrmse,
            'RMSE': rmse,
            'MAE': mae,
            'MSE': mse,
            'MAPE': mape
        })

    # Calculate overall metrics across all years
    y_test_total = results_df['y_test'].values
    y_pred_total = results_df['y_pred'].values

    mse_total = mean_squared_error(y_test_total, y_pred_total)
    rmse_total = np.sqrt(mse_total)
    mae_total = mean_absolute_error(y_test_total, y_pred_total)
    mape_total = np.mean(np.abs((y_test_total - y_pred_total) / y_test_total)) * 100
    rrmse_total = (rmse_total / np.mean(y_test_total)) * 100

    # Append the overall metrics to the list
    metrics_list.append({
        'year': 'total',
        'RRMSE': rrmse_total,
        'RMSE': rmse_total,
        'MAE': mae_total,
        'MSE': mse_total,
        'MAPE': mape_total
    })

    # Calculate the standard deviation of the metrics (excluding the overall metrics)
    metrics_list.append({
        'year': 'sd',
        'RRMSE': np.std([m['RRMSE'] for m in metrics_list[:-1]]),
        'RMSE': np.std([m['RMSE'] for m in metrics_list[:-1]]),
        'MAE': np.std([m['MAE'] for m in metrics_list[:-1]]),
        'MSE': np.std([m['MSE'] for m in metrics_list[:-1]]),
        'MAPE': np.std([m['MAPE'] for m in metrics_list[:-1]])
    })

    # Create a DataFrame to store the metrics
    metrics_df = pd.DataFrame(metrics_list)

    # Save the DataFrame to a CSV file
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    metrics_df.to_csv(output_csv_path, index=False)

    return metrics_df


def yield_prediction(estimator, params_path, model_name, metric_name, X_selected, y_selected, target_region, selected_years, result_save_path):
    os.makedirs(result_save_path, exist_ok=True)

    csv_path = f'{params_path}/{model_name}_best_params.csv'
    # Best Parameters 로드
    best_params = load_best_params(csv_path)
    print("Best Parameters:", best_params)

    # Leave-One-Year-Out 교차 검증을 통해 y_test와 y_pred를 얻음
    results_df = leave_one_year_out(X_selected, y_selected, estimator, best_params, target_region, selected_years)

    # Metrics 계산
    metrics_df = calculate_metrics_from_df(results_df, output_csv_path=f'{result_save_path}/{model_name}_metrics.csv')

    # Metrics 출력
    print("Metrics for each year:")
    print(metrics_df)
    print('-' * 65)
    print("Average RRMSE:", metrics_df.loc[metrics_df['year'] == 'total', 'RRMSE'].values[0])
    print("Average RMSE:", metrics_df.loc[metrics_df['year'] == 'total', 'RMSE'].values[0])
    print("Average MAE:", metrics_df.loc[metrics_df['year'] == 'total', 'MAE'].values[0])
    print("Average MSE:", metrics_df.loc[metrics_df['year'] == 'total', 'MSE'].values[0])
    print("Average MAPE:", metrics_df.loc[metrics_df['year'] == 'total', 'MAPE'].values[0])

    # 결과를 CSV 파일로 저장
    results_df.to_csv(f'{result_save_path}/{model_name}_results.csv', index=False)

    # 시각화: 연도별로 실제값, 예측값, 그리고 선택된 메트릭을 시각화
    plot_results(selected_years, results_df, target_region, metric_name=metric_name)

    # 시각화: 지역별로 평균 메트릭을 시각화
    plot_avg_metric_by_region(selected_years, results_df, target_region, metric_name=metric_name)

    # 전체 결과를 바탕으로 수확량 예측 시각화
    plot_yield_estimation(results_df)
