# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:27:28 2024

This script imports the data, applies data cleaning, renaming and smoothing. 
It returns the clean base dataset and reduced/indexed dataset.

@author: bianc
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression

def clean_data():
    os.chdir("C:/Users/bianc/Documents/Documents/OneDrive - University of Exeter/Master Thesis/Code")    
    ESG_dataset = pd.read_csv('ESG_data.csv')

  # Modify dataset
    # Rename columns 
    ESG_dataset.rename(columns={
        'Total Return (%)': 'return',
        'sector name': 'sectorName',
        'Market Cap ($m)': 'marketCap',
        'industry group name': 'indusGroup',
        'industry name': 'industry',
        'subindustry name': 'subindustry',
        'CountryCode': 'country'},
        inplace=True)
    
    # Format the categorical variables
    ESG_dataset = ESG_dataset[~ESG_dataset['sectorName'].isna()] # Filter out rows where sector_name is NaN
    ESG_dataset['country'] = ESG_dataset['country'].str.replace(r'\(.*?\)', '', regex=True).astype('category') # Remove brackets
    ESG_dataset['Year'] = pd.to_datetime(ESG_dataset['Year'], format='%Y') # Get years in date format
    ESG_dataset['sectorName'] = ESG_dataset['sectorName'].astype('category')
    
    # Exclude outlier case
    ESG_dataset = ESG_dataset[ESG_dataset['Identifier'] != 'ZP403275']
    # Clean for duplicates  
    numeric_column_indices = [2, 3, 4, 5, 6] 
    numeric_col = ESG_dataset.columns[numeric_column_indices].tolist()
    ESG_dataset_numeric_mean = ESG_dataset.groupby(['Year', 'Identifier'])[numeric_col].mean().reset_index()
    categoric_columns = [col for col in ESG_dataset.columns if col not in numeric_col]
    ESG_dataset_categoric = ESG_dataset[categoric_columns].drop_duplicates(['Year', 'Identifier'])
    ESG_dataset = pd.merge(ESG_dataset_categoric, ESG_dataset_numeric_mean, on=['Year', 'Identifier'])
    
    # Create new variables    
    ESG_dataset['ESGscore'] = ESG_dataset[['EnvironmentalScore', 'GovernanceScore', 'SocialScore']].mean(axis=1)

    # Define Region mapping
    country_to_region = {
        'AE': 'Middle East',
        'AR': 'Latin America',
        'AT': 'Europe',
        'AU': 'Oceania',
        'BE': 'Europe',
        'BR': 'Latin America',
        'CA': 'North America',
        'CH': 'Europe',
        'CL': 'Latin America',
        'CN': 'MED Asia',
        'CO': 'Latin America',
        'CZ': 'Europe',
        'DE': 'Europe',
        'DK': 'Europe',
        'EG': 'Africa',
        'ES': 'Europe',
        'FI': 'Europe',
        'FR': 'Europe',
        'GB': 'Europe',
        'GR': 'Europe',
        'HK': 'MED Asia',
        'HU': 'Europe',
        'ID': 'LED Asia',
        'IE': 'Europe',
        'IL': 'Middle East',
        'IN': 'MED Asia',
        'IT': 'Europe',
        'JO': 'Middle East',
        'JP': 'MED Asia',
        'KR': 'MED Asia',
        'KW': 'Middle East',
        'LU': 'Europe',
        'MA': 'Africa',
        'MX': 'Latin America',
        'MY': 'LED Asia',
        'NG': 'Africa',
        'NL': 'Europe',
        'NO': 'Europe',
        'NZ': 'Oceania',
        'PE': 'Latin America',
        'PH': 'LED Asia',
        'PK': 'LED Asia',
        'PL': 'Europe',
        'PT': 'Europe',
        'QA': 'Middle East',
        'RU': 'Europe',
        'SA': 'Middle East',
        'SE': 'Europe',
        'SG': 'MED Asia',
        'TH': 'LED Asia',
        'TR': 'Middle East',
        'TW': 'MED Asia',
        'US': 'North America',
        'VE': 'Latin America',
        'ZA': 'Africa'}

    ESG_dataset['Region'] = ESG_dataset['country'].map(country_to_region).astype('category')  
    
    # Developing and Emerging economies
    developed_country_codes = {
    'AT', 'BE', 'DK', 'FI', 'FR', 'DE', 'GR', 'IE', 'IT', 'LU',
    'NL', 'PT', 'ES', 'SE', 'GB', 'BG', 'HR', 'CY', 'CZ', 'EE',
    'HU', 'LV', 'LT', 'MT', 'PL', 'RO', 'SK', 'SI', 'IS', 'NO',
    'CH', 'AU', 'CA', 'JP', 'NZ', 'US', 'RU'}
    
    ESG_dataset['Development'] = ESG_dataset['country'].apply(
        lambda code: 'Developed' if code in developed_country_codes else 'Emerging').astype('category')
    #https://www.un.org/en/development/desa/policy/wesp/wesp_current/2014wesp_country_classification.pdf
    
  # Idiosyncratic Total Returns
    # Benchmark
    ESG_dataset['BenchmarkReturn'] = ESG_dataset.groupby('Year')['return'].transform('mean')
    # Beta for each Identifier
    betas_list = []
    identifiers = ESG_dataset['Identifier'].unique()
    for identifier in identifiers:
        df = ESG_dataset[ESG_dataset['Identifier'] == identifier]
        X = df[['BenchmarkReturn']].values.reshape(-1, 1)  # convert from 1D to 2D (n_samples, n_features) as required by sklearn 
        y = df['return'].values
        model = LinearRegression().fit(X, y)               # run Linear regressions to find beta
        beta = model.coef_[0]                              # extract Coefficient (beta)
        betas_list.append({'Identifier': identifier, 'Beta': beta})      
    betas = pd.DataFrame(betas_list) 
    ESG_dataset = ESG_dataset.merge(betas, on='Identifier', how='left')    
    ESG_dataset['IdioReturn'] = ESG_dataset['return'] - (ESG_dataset['BenchmarkReturn'] * ESG_dataset['Beta'])
    ESG_dataset.drop(columns=['BenchmarkReturn', 'Beta'], inplace=True)
    ESG_dataset['IdioReturn'] = ESG_dataset['IdioReturn'].clip(lower=-0.9999)
        
    # Apply log transformation to 'marketCap' and 'return' columns
    ESG_dataset['lnMarketCap'] = np.log(ESG_dataset['marketCap'] + 1 - ESG_dataset['marketCap'].min())
    ESG_dataset['lnIdioReturn'] = np.log(ESG_dataset['IdioReturn'] + 1 - ESG_dataset['IdioReturn'].min())
    # Set index
    ESG_dataset.set_index(['Identifier', 'Year'], inplace=True)
    
  # Create sub-datasets
    # Relevant variables nonlnMarketCap scaled
    df_column_indices = [0, 9, 10,  2, 4, 6, 5, 8, 13, 12] 
    df_ESG = ESG_dataset.iloc[:, df_column_indices] 
    
    # Apply Normalisation
    numeric_col_indices = [4, 5, 6, 7, 8, 9] 
    df_numeric_std = df_ESG.iloc[:,numeric_col_indices]
   # scaling = QuantileTransformer(output_distribution='normal')      
   # scaling = PowerTransformer(method="yeo-johnson")
   # scaling = RobustScaler(quantile_range=(25, 75))
    scaling = StandardScaler()
    df_numeric_std = pd.DataFrame(scaling.fit_transform(df_ESG.iloc[:, numeric_col_indices]), columns=df_ESG.columns[numeric_col_indices], index=df_ESG.index)
    df_ESG_std = pd.concat([df_numeric_std, df_ESG[['country', 'Region', 'Development', 'sectorName']]], axis=1)
    
    return ESG_dataset, df_ESG, df_numeric_std, df_ESG_std
   
ESG_dataset, df_ESG, df_numeric_std, df_ESG_std = clean_data()

ESG_dataset.info()
ESG_dataset.head()

df_ESG.info()
df_ESG.head()

df_numeric_std.info()
df_numeric_std.head()

df_ESG_std.info()
df_ESG_std.head()

