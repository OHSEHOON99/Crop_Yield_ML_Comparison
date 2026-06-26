import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 최적의 하이퍼파라미터를 로드하는 함수
def load_best_params(csv_path):
    # CSV 파일 로드
    df = pd.read_csv(csv_path)
    
    # 첫 번째 행을 딕셔너리로 변환
    best_params = df.iloc[0].to_dict()
    
    # 값의 형식을 적절히 변환
    for key, value in best_params.items():
        if pd.isna(value):  # NaN 값을 None으로 변환
            best_params[key] = None
        elif isinstance(value, float) and value.is_integer():  # 정수형 실수를 정수로 변환
            best_params[key] = int(value)
        else:
            best_params[key] = value
    
    return best_params


def plot_results(selected_years, results_df, region_names, metric_name):
    n_years = len(selected_years)
    n_regions = len(region_names)
    
    # Calculate the number of rows and columns for subplots
    n_cols = 2
    n_rows = (n_years + n_cols - 1) // n_cols
    
    # Calculate the maximum value for the selected metric to fix y-axis across subplots
    max_metric_value = 0
    for test_year in selected_years:
        year_results = results_df[results_df['year'] == test_year]
        if year_results.empty:
            continue

        y_test_means = year_results.groupby('region_name')['y_test'].mean().reindex(region_names, fill_value=0).values
        y_pred_means = year_results.groupby('region_name')['y_pred'].mean().reindex(region_names, fill_value=0).values
        
        if metric_name == 'RRMSE':
            metric_values = np.sqrt(((y_pred_means - y_test_means) ** 2)) / y_test_means * 100
        elif metric_name == 'RMSE':
            metric_values = np.sqrt(((y_pred_means - y_test_means) ** 2))
        elif metric_name == 'MAE':
            metric_values = np.abs(y_pred_means - y_test_means)
        elif metric_name == 'MSE':
            metric_values = ((y_pred_means - y_test_means) ** 2)
        elif metric_name == 'MAPE':
            metric_values = (np.abs((y_test_means - y_pred_means) / y_test_means)) * 100
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")
        
        max_metric_value = max(max_metric_value, np.nanmax(metric_values)) + 1  # Handle NaN values safely

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows), sharex=True)
    axes = axes.flatten()

    for i, (ax, test_year) in enumerate(zip(axes, selected_years)):
        # Filter the results for the current year
        year_results = results_df[results_df['year'] == test_year]
        
        if year_results.empty:
            ax.set_visible(False)  # Hide axes with no data
            continue
        
        # Get the mean values for y_test and y_pred
        y_test_means = year_results.groupby('region_name')['y_test'].mean().reindex(region_names, fill_value=0).values
        y_pred_means = year_results.groupby('region_name')['y_pred'].mean().reindex(region_names, fill_value=0).values
        
        # Calculate the selected metric for each region
        if metric_name == 'RRMSE':
            metric_means = np.sqrt(((y_pred_means - y_test_means) ** 2)) / y_test_means * 100
        elif metric_name == 'RMSE':
            metric_means = np.sqrt(((y_pred_means - y_test_means) ** 2))
        elif metric_name == 'MAE':
            metric_means = np.abs(y_pred_means - y_test_means)
        elif metric_name == 'MSE':
            metric_means = ((y_pred_means - y_test_means) ** 2)
        elif metric_name == 'MAPE':
            metric_means = (np.abs((y_test_means - y_pred_means) / y_test_means)) * 100
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")
        
        x = np.arange(n_regions)
        bar_width = 0.35
        
        # Plot the actual and predicted values as bar plots
        ax.bar(x - bar_width/2, y_test_means, bar_width, label='True Values')
        ax.bar(x + bar_width/2, y_pred_means, bar_width, label='Predicted Values')
        
        # Plot the metric as a line plot
        ax2 = ax.twinx()
        ax2.plot(x, metric_means, color='r', marker='o', label=metric_name)
        
        # Set titles and labels
        ax.set_title(f'Year {test_year}')
        ax.set_xlabel('Region')
        ax.set_ylabel('Value')
        ax2.set_ylabel(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(region_names, rotation=90)
        
        # Set the same y-axis limits across all subplots for the metric
        ax2.set_ylim(0, max_metric_value)
        
        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    
def plot_avg_metric_by_region(selected_years, results_df, region_names, metric_name=None):
    # `metric_name`을 명시적으로 전달하도록 하고, 기본값을 None으로 설정
    if metric_name is None:
        raise ValueError("metric_name must be provided")
    
    # 특정 메트릭에 대한 정보를 계산하여 지역별로 평균 계산
    avg_metric_by_region = []
    for region in region_names:
        region_results = results_df[(results_df['region_name'] == region) & (results_df['year'].isin(selected_years))]
        y_test_means = region_results['y_test'].values
        y_pred_means = region_results['y_pred'].values
        
        if metric_name == 'RRMSE':
            metric_values = np.sqrt(((y_pred_means - y_test_means) ** 2)) / y_test_means * 100
        elif metric_name == 'RMSE':
            metric_values = np.sqrt(((y_pred_means - y_test_means) ** 2))
        elif metric_name == 'MAE':
            metric_values = np.abs(y_pred_means - y_test_means)
        elif metric_name == 'MSE':
            metric_values = ((y_pred_means - y_test_means) ** 2)
        elif metric_name == 'MAPE':
            metric_values = (np.abs((y_test_means - y_pred_means) / y_test_means)) * 100
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")
        
        avg_metric_by_region.append(np.mean(metric_values))
    
    # 시각화를 위한 준비
    x = np.arange(len(region_names))
    plt.figure(figsize=(12, 8))
    
    # 막대 그래프 생성
    plt.bar(x, avg_metric_by_region, color='skyblue')
    
    # 그래프 레이블 및 제목 설정
    plt.xlabel('Region')
    plt.ylabel(f'Average {metric_name}')
    plt.title(f'Average {metric_name} by Region for Selected Years: {selected_years}')
    plt.xticks(x, region_names, rotation=90)
    
    # 그래프 보여주기
    plt.tight_layout()
    plt.show()


def plot_yield_estimation(results_df):
    # y_test와 y_pred가 리스트나 문자열로 저장되어 있다면 float으로 변환
    results_df['y_test'] = results_df['y_test'].apply(lambda x: float(x) if isinstance(x, (list, str)) else x)
    results_df['y_pred'] = results_df['y_pred'].apply(lambda x: float(x) if isinstance(x, (list, str)) else x)

    # 연도별 y_test와 y_pred의 평균 계산
    mean_yields_by_year = results_df.groupby('year')[['y_test', 'y_pred']].mean().reset_index()

    # 그래프 생성
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # 실제 수확량과 예측 수확량을 막대 그래프로 시각화
    width = 0.4
    ax1.bar(mean_yields_by_year['year'] - width/2, mean_yields_by_year['y_test'], width=width, align='center', label='Actual Yield')
    ax1.bar(mean_yields_by_year['year'] + width/2, mean_yields_by_year['y_pred'], width=width, align='center', label='Predicted Yield')

    # x축 설정
    ax1.set_xticks(mean_yields_by_year['year'])
    ax1.set_xticklabels(mean_yields_by_year['year'].astype(int))

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Yield (centner/ha)')
    ax1.tick_params(axis='y')

    # 범례 추가
    ax1.legend(loc='upper left')

    plt.title('Ukraine Wheat Yield Estimation')
    plt.show()


# Function to visualize results with y=x trend line and correlation only
def scatterplot_visualization(df, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(df['y_test'], df['y_pred'], alpha=0.5)
    
    # y=x 직선 추가 (범례 없이)
    max_val = max(df['y_test'].max(), df['y_pred'].max()) + 10
    plt.plot([0, max_val], [0, max_val], color='green', linestyle='--')
    
    # 축 범위 설정 (최소값 0)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    
    plt.xlabel('True Values (y_test)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.title(f'Scatter Plot of {title} Yield Estimation')
    
    # 상관관계 계산
    correlation = df['y_test'].corr(df['y_pred'])
    
    # 상관관계 출력
    plt.text(0, max_val * 0.95, f'Correlation: {correlation:.2f}', color='red', fontsize=12)
    
    plt.grid(True)
    plt.show()


def plot_metric_by_algorithm(directory, metric):
    # Initialize a dictionary to hold the results for each algorithm
    results = {}
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('_metrics.csv'):
            # Extract the algorithm name from the filename
            algorithm_name = filename.split('_metrics.csv')[0]
            
            # Load the CSV file
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            
            # Exclude 'total' and 'sd' rows
            df = df[~df['year'].isin(['total', 'sd'])]
            
            # Store the metric values by year in the results dictionary
            results[algorithm_name] = {
                'metric_by_year': dict(zip(df['year'], df[metric]))
            }
    
    # Plot settings
    plt.figure(figsize=(12, 8))
    
    # Generate a sorted list of years from the first algorithm's data
    years = sorted(list(df['year'].unique()))
    
    # Plot the metric for each algorithm
    for algorithm_name, result in results.items():
        metric_by_year = result['metric_by_year']
        metric_values = [metric_by_year.get(year, None) for year in years]  # Extract metric values for each year
        plt.plot(years, metric_values, marker='o', label=algorithm_name)
    
    # Graph labels
    plt.xlabel('Year')
    plt.ylabel(metric)
    plt.title(f'{metric} for Each Year by Algorithm')
    plt.legend()
    plt.grid(True)
    
    # Show the graph
    plt.show()
