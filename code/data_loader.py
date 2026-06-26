import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class CropYieldDataLoader:
    def __init__(self, target_region, combined_file_path, crop_yield_file_path):
        self.target_region = sorted(target_region)
        self.combined_file_path = combined_file_path
        self.crop_yield_file_path = crop_yield_file_path
        self.n_regions = len(self.target_region)

        self.X = None
        self.y = None
        self.X_selected = None
        self.y_selected = None
        self.all_years = None
        self.selected_years = None

    def load_data(self):
        # Load and process combined data
        combined_df = pd.read_csv(self.combined_file_path)
        available_regions = combined_df['region_name'].unique()
        self.target_region = [region for region in self.target_region if region in available_regions]
        self.n_regions = len(self.target_region)

        self.all_years = sorted(combined_df['year'].unique())

        era5_land_indicators = ['t2m', 'stl1', 'stl2', 'ssr', 'tp', 'swvl1', 'swvl2']
        modis_indicators = ['NDVI', 'EVI', 'LST_Day', 'LST_Night', 'LAI', 'FPAR']

        available_indicators = combined_df.columns.difference(['region_name', 'year', 'month'])

        era5_land_indicators = sorted([ind for ind in era5_land_indicators if ind in available_indicators])
        modis_indicators = sorted([ind for ind in modis_indicators if ind in available_indicators])

        ordered_indicators = era5_land_indicators + modis_indicators
        print("Indicators:")
        for ind in ordered_indicators:
            print(f"- {ind}")

        # Normalize the data (excluding 'region_name', 'year', 'month')
        scaler = MinMaxScaler()
        combined_df[ordered_indicators] = scaler.fit_transform(combined_df[ordered_indicators])

        # Load crop yield data
        crop_yield_df = pd.read_csv(self.crop_yield_file_path)
        self.X, missing_data = self.process_data_with_debug(combined_df, ordered_indicators)

        # Process crop yield data
        self.y = self.process_crop_yield_data(crop_yield_df, self.X)

    def process_data_with_debug(self, df, ordered_indicators):
        processed_data = []
        missing_data = []

        for year in sorted(df['year'].unique()):
            year_df = df[df['year'] == year]
            year_df = year_df.set_index('region_name').loc[self.target_region].reset_index()

            for region in self.target_region:
                region_yearly_data = year_df[year_df['region_name'] == region]
                if len(region_yearly_data) < 8:  # Ensure all 8 months of data are present
                    missing_data.append((year, region, region_yearly_data['month'].tolist()))
                    continue
                features = region_yearly_data[ordered_indicators].values
                if len(features) != 8:  # Ensure all 8 months of data are present
                    missing_data.append((year, region, region_yearly_data['month'].tolist()))
                    continue
                processed_data.append(features.flatten())  # Flatten all features for 8 months

        if missing_data:
            print("Missing data details (year, region, missing_months):")
            for item in missing_data:
                print(item)
        else:
            print("No missing data.")

        return np.array(processed_data), missing_data

    def process_crop_yield_data(self, crop_yield_df, X):
        crop_id_211_df = crop_yield_df[crop_yield_df['crop_id'] == 211]
        sorted_crop_yield_df = pd.DataFrame()

        for year in sorted(crop_id_211_df['year'].unique()):
            year_df = crop_id_211_df[crop_id_211_df['year'] == year]
            year_df = year_df.set_index('region_name').loc[self.target_region].reset_index()
            sorted_crop_yield_df = pd.concat([sorted_crop_yield_df, year_df])

        sorted_crop_yield_df = sorted_crop_yield_df[['region_name', 'year', 'yield']]
        y = sorted_crop_yield_df['yield'].values
        y = y[:X.shape[0]]
        # print(f"All years: {self.all_years}")
        # print("Processed Data Shape (X):", X.shape)
        # print("Normalized Crop Yield Shape (y):", y.shape)

        return y

    def filter_years(self, selected_years):
        """Filter loaded arrays to the selected years."""
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Please run load_data() first.")
        
        self.selected_years = selected_years
        
        expanded_years = np.repeat(self.all_years, self.n_regions)
        selected_indices = np.isin(expanded_years, self.selected_years)
        self.X_selected = self.X[selected_indices]
        self.y_selected = self.y[selected_indices]

        print(f"Selected years: {self.selected_years}")
        print("Filtered Data Shape (X_selected):", self.X_selected.shape)
        print("Filtered Data Shape (y_selected):", self.y_selected.shape)

        return self.X_selected, self.y_selected
