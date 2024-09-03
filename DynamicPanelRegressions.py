# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:26:59 2024

@author: bianc

This script is for the Dynamic Panel Data Models employed in H2, H3 and H4.
"""

#%% Quantile Regression

import pandas as pd
import numpy as np
from  pydynpd import regression

from DataProcessing import clean_data, reset_index
from ClusteringModule import cluster_dataframe

ESG_dataset, df_ESG, df_numeric_std, df_ESG_std = clean_data()
df_clust_std, df_cluster0, df_cluster1, df_cluster2, df_cluster3 = cluster_dataframe()
df_ESG_R = reset_index()


#%% Defining Optimal Model
# https://pypi.org/project/pydynpd/0.1.3/
# https://github.com/dazhwu/pydynpd/blob/main/README.md
# https://github.com/dazhwu/pydynpd/blob/main/vignettes/API.md how to extract the variables
df_ESG_std.info()
df = df_ESG_std.reset_index()
df.info()


# Used to determine the optimum lag model
# command_str='lnIdioReturn L(1:?).lnIdioReturn L(1:?).EnvironmentalScore L(1:?).SocialScore L(1:?).GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:10) iv(lnMarketCap) | collapse'

# Confirmed best model
command_str='lnIdioReturn L(1:4).lnIdioReturn EnvironmentalScore L1.EnvironmentalScore SocialScore L1.SocialScore GovernanceScore L1.GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:10) iv(lnMarketCap) | collapse '
mydpd = regression.abond(command_str, df, ['Identifier', 'Year'])


# Extracting overall information
m = mydpd.models[0]
num_obs = m.num_obs
num_groups = m.N
hansen_p_value = m.hansen.p_value
ar1_p_value = m.AR_list[0].P_value
ar2_p_value = m.AR_list[1].P_value

# Extracting coefficients and p-values
regression_table = m.regression_table

# Filter significant variables
significant_vars = regression_table[regression_table['sig'] != ''].copy()
significant_vars = significant_vars[['variable', 'coefficient', 'p_value']]

# Create DataFrame for summary
dpm_gen = {
    'Number of Observations': [num_obs],
    'Number of Groups': [num_groups],
    'Hansen Test p-value': [hansen_p_value],
    'Arellano-Bond AR(1) p-value': [ar1_p_value],
    'Arellano-Bond AR(2) p-value': [ar2_p_value],
}

# Add coefficients and p-values for significant variables
for index, row in significant_vars.iterrows():
    dpm_gen[f'Coef_{row["variable"]}'] = [row['coefficient']]
    dpm_gen[f'P-Value_{row["variable"]}'] = [row['p_value']]

results_dpm_gen = pd.DataFrame(dpm_gen)

#%% H3: per Region

# Latin America
df = df_ESG_std[df_ESG_std['Region'] == 'Latin America'].reset_index()
command_str='lnIdioReturn L(1:3).lnIdioReturn EnvironmentalScore L(1:3).EnvironmentalScore SocialScore L(1:3).SocialScore GovernanceScore L(1:3).GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:7) iv(lnMarketCap) | collapse'

def extract_results_DPM():
    m = mydpd.models[0]
    num_obs = m.num_obs
    num_groups = m.N
    hansen_p_value = m.hansen.p_value
    ar1_p_value = m.AR_list[0].P_value
    ar2_p_value = m.AR_list[1].P_value
    
    # Extracting coefficients and p-values
    regression_table = m.regression_table
    significant_vars = regression_table[regression_table['sig'] != ''].copy()
    significant_vars = significant_vars[['variable', 'coefficient', 'p_value', 'sig']]
    # Create DataFrame for summary
    dpm_result = {
        'Region' : 'Latin American',
        'Number of Observations': [num_obs],
        'Number of Groups': [num_groups],
        'Number of Instruments': '30',
        'Hansen Test p-value': [hansen_p_value],
        'Arellano-Bond AR(1) p-value': [ar1_p_value],
        'Arellano-Bond AR(2) p-value': [ar2_p_value],
    }
    
    # Add coefficients and p-values for significant variables
    for index, row in significant_vars.iterrows():
        p_value_with_sig = f"{row['p_value']:.3e} ({row['sig']})" if pd.notna(row['sig']) else f"{row['p_value']:.3e}"
        dpm_result[f'Coef_{row["variable"]}'] = row['coefficient']
        dpm_result[f'P-Value_{row["variable"]}'] = p_value_with_sig
    
    results_dpm_Latin = pd.DataFrame(dpm_result)

    return results_dpm_Latin
results_dpm_Latin = extract_results_DPM()

#%% Africa
df = df_ESG_std[df_ESG_std['Region'] == 'Africa'].reset_index()
command_str='lnIdioReturn L(1:1).lnIdioReturn EnvironmentalScore L1.EnvironmentalScore SocialScore L1.SocialScore GovernanceScore L1.GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:10) iv(lnMarketCap) | collapse '

def extract_results_DPM():
    m = mydpd.models[0]
    num_obs = m.num_obs
    num_groups = m.N
    hansen_p_value = m.hansen.p_value
    ar1_p_value = m.AR_list[0].P_value
    ar2_p_value = m.AR_list[1].P_value
    
    # Extracting coefficients and p-values
    regression_table = m.regression_table
    significant_vars = regression_table[regression_table['sig'] != ''].copy()
    significant_vars = significant_vars[['variable', 'coefficient', 'p_value', 'sig']]
    # Create DataFrame for summary
    dpm_result = {
        'Region' : 'Africa',
        'Number of Observations': [num_obs],
        'Number of Groups': [num_groups],
        'Number of Instruments': '42',
        'Hansen Test p-value': [hansen_p_value],
        'Arellano-Bond AR(1) p-value': [ar1_p_value],
        'Arellano-Bond AR(2) p-value': [ar2_p_value],
    }
    
    # Add coefficients and p-values for significant variables
    for index, row in significant_vars.iterrows():
        p_value_with_sig = f"{row['p_value']:.3e} ({row['sig']})" if pd.notna(row['sig']) else f"{row['p_value']:.3e}"
        dpm_result[f'Coef_{row["variable"]}'] = row['coefficient']
        dpm_result[f'P-Value_{row["variable"]}'] = p_value_with_sig
    
    results_dpm_Africa = pd.DataFrame(dpm_result)

    return results_dpm_Africa
results_dpm_Africa = extract_results_DPM()


# Europe
df = df_ESG_std[df_ESG_std['Region'] == 'Europe'].reset_index()
command_str='lnIdioReturn L(1:1).lnIdioReturn EnvironmentalScore L(1:2).EnvironmentalScore SocialScore L(1:2).SocialScore GovernanceScore L(1:2).GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:10) iv(lnMarketCap) | collapse'

def extract_results_DPM():
    m = mydpd.models[0]
    num_obs = m.num_obs
    num_groups = m.N
    hansen_p_value = m.hansen.p_value
    ar1_p_value = m.AR_list[0].P_value
    ar2_p_value = m.AR_list[1].P_value
    
    # Extracting coefficients and p-values
    regression_table = m.regression_table
    significant_vars = regression_table[regression_table['sig'] != ''].copy()
    significant_vars = significant_vars[['variable', 'coefficient', 'p_value', 'sig']]
    # Create DataFrame for summary
    dpm_result = {
        'Region' : 'Europe',
        'Number of Observations': [num_obs],
        'Number of Groups': [num_groups],
        'Number of Instruments': '42',
        'Hansen Test p-value': [hansen_p_value],
        'Arellano-Bond AR(1) p-value': [ar1_p_value],
        'Arellano-Bond AR(2) p-value': [ar2_p_value],
    }
    
    # Add coefficients and p-values for significant variables
    for index, row in significant_vars.iterrows():
        p_value_with_sig = f"{row['p_value']:.3e} ({row['sig']})" if pd.notna(row['sig']) else f"{row['p_value']:.3e}"
        dpm_result[f'Coef_{row["variable"]}'] = row['coefficient']
        dpm_result[f'P-Value_{row["variable"]}'] = p_value_with_sig
    
    results_dpm_Europe = pd.DataFrame(dpm_result)

    return results_dpm_Europe

results_dpm_Europe = extract_results_DPM()

# LED Asia
df = df_ESG_std[df_ESG_std['Region'] == 'LED Asia'].reset_index()
command_str='lnIdioReturn L(1:1).lnIdioReturn EnvironmentalScore L1.EnvironmentalScore SocialScore L1.SocialScore GovernanceScore L1.GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:10) iv(lnMarketCap) | collapse '

def extract_results_DPM():
    m = mydpd.models[0]
    num_obs = m.num_obs
    num_groups = m.N
    hansen_p_value = m.hansen.p_value
    ar1_p_value = m.AR_list[0].P_value
    ar2_p_value = m.AR_list[1].P_value
    
    # Extracting coefficients and p-values
    regression_table = m.regression_table
    significant_vars = regression_table[regression_table['sig'] != ''].copy()
    significant_vars = significant_vars[['variable', 'coefficient', 'p_value', 'sig']]
    # Create DataFrame for summary
    dpm_result = {
        'Region' : 'LED Asia',
        'Number of Observations': [num_obs],
        'Number of Groups': [num_groups],
        'Number of Instruments': '42',
        'Hansen Test p-value': [hansen_p_value],
        'Arellano-Bond AR(1) p-value': [ar1_p_value],
        'Arellano-Bond AR(2) p-value': [ar2_p_value],
    }
    
    # Add coefficients and p-values for significant variables
    for index, row in significant_vars.iterrows():
        p_value_with_sig = f"{row['p_value']:.3e} ({row['sig']})" if pd.notna(row['sig']) else f"{row['p_value']:.3e}"
        dpm_result[f'Coef_{row["variable"]}'] = row['coefficient']
        dpm_result[f'P-Value_{row["variable"]}'] = p_value_with_sig
    
    results_dpm_LED = pd.DataFrame(dpm_result)

    return results_dpm_LED

results_dpm_LED = extract_results_DPM()


# MED Asia

df = df_ESG_std[df_ESG_std['Region'] == 'MED Asia'].reset_index()
command_str='lnIdioReturn L(1:1).lnIdioReturn EnvironmentalScore L(1:2).EnvironmentalScore SocialScore L(1:2).SocialScore GovernanceScore L(1:2).GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:10) iv(lnMarketCap) | collapse'

def extract_results_DPM():
    m = mydpd.models[0]
    num_obs = m.num_obs
    num_groups = m.N
    hansen_p_value = m.hansen.p_value
    ar1_p_value = m.AR_list[0].P_value
    ar2_p_value = m.AR_list[1].P_value
    
    # Extracting coefficients and p-values
    regression_table = m.regression_table
    significant_vars = regression_table[regression_table['sig'] != ''].copy()
    significant_vars = significant_vars[['variable', 'coefficient', 'p_value', 'sig']]
    # Create DataFrame for summary
    dpm_result = {
        'Region' : 'MED Asia',
        'Number of Observations': [num_obs],
        'Number of Groups': [num_groups],
        'Number of Instruments': '42',
        'Hansen Test p-value': [hansen_p_value],
        'Arellano-Bond AR(1) p-value': [ar1_p_value],
        'Arellano-Bond AR(2) p-value': [ar2_p_value],
    }
    
    # Add coefficients and p-values for significant variables
    for index, row in significant_vars.iterrows():
        p_value_with_sig = f"{row['p_value']:.3e} ({row['sig']})" if pd.notna(row['sig']) else f"{row['p_value']:.3e}"
        dpm_result[f'Coef_{row["variable"]}'] = row['coefficient']
        dpm_result[f'P-Value_{row["variable"]}'] = p_value_with_sig
    
    results_dpm_MED = pd.DataFrame(dpm_result)

    return results_dpm_MED

results_dpm_MED = extract_results_DPM()

# Middle East
df = df_ESG_std[df_ESG_std['Region'] == 'Middle East'].reset_index()

# North America
df = df_ESG_std[df_ESG_std['Region'] == 'North America'].reset_index()
command_str='lnIdioReturn L(1:3).lnIdioReturn EnvironmentalScore L(1:2).EnvironmentalScore SocialScore L(1:2).SocialScore GovernanceScore L(1:2).GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:10) iv(lnMarketCap) | collapse'

def extract_results_DPM():
    m = mydpd.models[0]
    num_obs = m.num_obs
    num_groups = m.N
    hansen_p_value = m.hansen.p_value
    ar1_p_value = m.AR_list[0].P_value
    ar2_p_value = m.AR_list[1].P_value
    
    # Extracting coefficients and p-values
    regression_table = m.regression_table
    significant_vars = regression_table[regression_table['sig'] != ''].copy()
    significant_vars = significant_vars[['variable', 'coefficient', 'p_value', 'sig']]
    # Create DataFrame for summary
    dpm_result = {
        'Region' : 'North America',
        'Number of Observations': [num_obs],
        'Number of Groups': [num_groups],
        'Number of Instruments': '42',
        'Hansen Test p-value': [hansen_p_value],
        'Arellano-Bond AR(1) p-value': [ar1_p_value],
        'Arellano-Bond AR(2) p-value': [ar2_p_value],
    }
    
    # Add coefficients and p-values for significant variables
    for index, row in significant_vars.iterrows():
        p_value_with_sig = f"{row['p_value']:.3e} ({row['sig']})" if pd.notna(row['sig']) else f"{row['p_value']:.3e}"
        dpm_result[f'Coef_{row["variable"]}'] = row['coefficient']
        dpm_result[f'P-Value_{row["variable"]}'] = p_value_with_sig
    
    results_dpm_US = pd.DataFrame(dpm_result)

    return results_dpm_US

results_dpm_US = extract_results_DPM()

# Oceania
df = df_ESG_std[df_ESG_std['Region'] == 'Oceania'].reset_index()
command_str='lnIdioReturn L(1:1).lnIdioReturn EnvironmentalScore L1.EnvironmentalScore SocialScore L1.SocialScore GovernanceScore L1.GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:10) iv(lnMarketCap) | collapse '

def extract_results_DPM():
    m = mydpd.models[0]
    num_obs = m.num_obs
    num_groups = m.N
    hansen_p_value = m.hansen.p_value
    ar1_p_value = m.AR_list[0].P_value
    ar2_p_value = m.AR_list[1].P_value
    
    # Extracting coefficients and p-values
    regression_table = m.regression_table
    significant_vars = regression_table[regression_table['sig'] != ''].copy()
    significant_vars = significant_vars[['variable', 'coefficient', 'p_value', 'sig']]
    # Create DataFrame for summary
    dpm_result = {
        'Region' : 'Oceania',
        'Number of Observations': [num_obs],
        'Number of Groups': [num_groups],
        'Number of Instruments': '42',
        'Hansen Test p-value': [hansen_p_value],
        'Arellano-Bond AR(1) p-value': [ar1_p_value],
        'Arellano-Bond AR(2) p-value': [ar2_p_value],
    }
    
    # Add coefficients and p-values for significant variables
    for index, row in significant_vars.iterrows():
        p_value_with_sig = f"{row['p_value']:.3e} ({row['sig']})" if pd.notna(row['sig']) else f"{row['p_value']:.3e}"
        dpm_result[f'Coef_{row["variable"]}'] = row['coefficient']
        dpm_result[f'P-Value_{row["variable"]}'] = p_value_with_sig
    
    results_dpm_Oceania = pd.DataFrame(dpm_result)

    return results_dpm_Oceania

results_dpm_Oceania = extract_results_DPM()


# Join results
result_df_Region = pd.concat([results_dpm_Africa, results_dpm_Europe, 
                          results_dpm_Oceania, results_dpm_LED, 
                          results_dpm_MED, results_dpm_Latin, 
                          results_dpm_US], axis=0)
result_df_Region.info()
result_df_Region.to_csv('result_df_Region.csv')      
    
#%% H3: per Development

# Get unique region names
df_emerg = df_ESG_std[df_ESG_std['Development'] == 'Emerging']
df = df_emerg.reset_index()


df_dev = df_ESG_std[df_ESG_std['Development'] == 'Developed']
df = df_dev.reset_index()

command_str='lnIdioReturn L(1:2).lnIdioReturn EnvironmentalScore L(1:2).EnvironmentalScore SocialScore L(1:2).SocialScore GovernanceScore L(1:2).GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:7) iv(lnMarketCap) | collapse'
mydpd = regression.abond(command_str, df, ['Identifier', 'Year'])


## Extract regression values

result_dpm_region = pd.DataFrame()

# Get unique region names
df = df_ESG_std.reset_index()
unique_region = df['Development'].unique()
# Loop over each unique Region
for dev in unique_region:
    # Filter the DataFrame for the current Region
    df_region = df[df['Development'] == dev]
    # Run the dynamic panel data model on the filtered DataFrame
    mydpd = regression.abond(command_str, df_region, ['Identifier', 'Year'])
    m = mydpd.models[0]

    # Extract overall information
    num_obs = m.num_obs
    num_groups = m.N
    hansen_p_value = m.hansen.p_value
    ar1_p_value = m.AR_list[0].P_value
    ar2_p_value = m.AR_list[1].P_value
    # Extract coefficients and p-values
    regression_table = m.regression_table
    significant_vars = regression_table[regression_table['sig'].isin(['*', '**', '***'])].copy()
    significant_vars = significant_vars[['variable', 'coefficient', 'p_value', 'sig']]
    
    # Create a dictionary for the current region
    dpm_region = {
        'Development Status': dev,
        'Number of Observations': num_obs,
        'Number of Groups': num_groups,
        'Hansen Test p-value': hansen_p_value,
        'Arellano-Bond AR(1) p-value': ar1_p_value,
        'Arellano-Bond AR(2) p-value': ar2_p_value,
    }
    
    # Add coefficients and p-values for significant variables
    for index, row in significant_vars.iterrows():
        # Combine p-value and significance mark into a single string
        p_value_with_sig = f"{row['p_value']:.3e} ({row['sig']})" if pd.notna(row['sig']) else f"{row['p_value']:.3e}"
        dpm_region[f'Coef_{row["variable"]}'] = row['coefficient']
        dpm_region[f'P-Value_{row["variable"]}'] = p_value_with_sig
    
    # Convert the dictionary to a DataFrame and append to result_dpm__region
    region_df = pd.DataFrame([dpm_region])
    result_dpm_region = pd.concat([result_dpm_region, region_df], ignore_index=True)



#%% H4: per Industry

# Defining the optimal model
df = df_ESG_std.reset_index()
unique_sectors = df['sectorName'].unique()

# Datasets for each unique sector

df = df_ESG_std[df_ESG_std['sectorName'] == 'Energy'].reset_index()
#command_str='lnIdioReturn L(1:4).lnIdioReturn EnvironmentalScore L1.EnvironmentalScore SocialScore L1.SocialScore GovernanceScore L1.GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:10) iv(lnMarketCap) | collapse '

df = df_ESG_std[df_ESG_std['sectorName'] == 'Consumer Discretionary'].reset_index()
#command_str='lnIdioReturn L(1:4).lnIdioReturn EnvironmentalScore L1.EnvironmentalScore SocialScore L1.SocialScore GovernanceScore L1.GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:10) iv(lnMarketCap) | collapse '

df = df_ESG_std[df_ESG_std['sectorName'] == 'Utilities'].reset_index()
#command_str='lnIdioReturn L(1:4).lnIdioReturn EnvironmentalScore L(1:3).EnvironmentalScore SocialScore L(1:3).SocialScore GovernanceScore L(1:3).GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:10) iv(lnMarketCap) | collapse'

df = df_ESG_std[df_ESG_std['sectorName'] == 'Materials'].reset_index()
#command_str='lnIdioReturn L(1:9).lnIdioReturn EnvironmentalScore L(1:2).EnvironmentalScore SocialScore L(1:2).SocialScore GovernanceScore L(1:2).GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:10) iv(lnMarketCap) | collapse'

df = df_ESG_std[df_ESG_std['sectorName'] == 'Financials'].reset_index()
#command_str='lnIdioReturn L(1:4).lnIdioReturn EnvironmentalScore L1.EnvironmentalScore SocialScore L1.SocialScore GovernanceScore L1.GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:7) iv(lnMarketCap) | collapse '

df = df_ESG_std[df_ESG_std['sectorName'] == 'Industrials'].reset_index()
#command_str='lnIdioReturn L(1:1).lnIdioReturn EnvironmentalScore L1.EnvironmentalScore SocialScore L1.SocialScore GovernanceScore L1.GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:7) iv(lnMarketCap) | collapse '

df = df_ESG_std[df_ESG_std['sectorName'] == 'Information Technology'].reset_index()
#command_str='lnIdioReturn L(1:1).lnIdioReturn EnvironmentalScore L1.EnvironmentalScore SocialScore L1.SocialScore GovernanceScore L1.GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:5) iv(lnMarketCap) | collapse '

df = df_ESG_std[df_ESG_std['sectorName'] == 'Consumer Staples'].reset_index()
#command_str='lnIdioReturn L(1:1).lnIdioReturn EnvironmentalScore L1.EnvironmentalScore SocialScore L1.SocialScore GovernanceScore L1.GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:10) iv(lnMarketCap) | collapse '

df  = df_ESG_std[df_ESG_std['sectorName'] == 'Real Estate'].reset_index()
#command_str='lnIdioReturn L(1:7).lnIdioReturn EnvironmentalScore L(1:2).EnvironmentalScore SocialScore L(1:2).SocialScore GovernanceScore L(1:2).GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:5) iv(lnMarketCap) | collapse'

df = df_ESG_std[df_ESG_std['sectorName'] == 'Communication Services'].reset_index()
#command_str='lnIdioReturn L(1:1).lnIdioReturn EnvironmentalScore L1.EnvironmentalScore SocialScore L1.SocialScore GovernanceScore L1.GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:10) iv(lnMarketCap) | collapse '

df = df_ESG_std[df_ESG_std['sectorName'] == 'Health Care'].reset_index()
#command_str='lnIdioReturn L(1:2).lnIdioReturn EnvironmentalScore L1.EnvironmentalScore SocialScore L1.SocialScore GovernanceScore L1.GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:5) iv(lnMarketCap) | collapse '

# Extracting single model information

# Extracting overall information
m = mydpd.models[0]
num_obs = m.num_obs
num_groups = m.N
hansen_p_value = m.hansen.p_value
ar1_p_value = m.AR_list[0].P_value
ar2_p_value = m.AR_list[1].P_value

# Extracting coefficients and p-values
regression_table = m.regression_table
significant_vars = regression_table[regression_table['sig'] != ''].copy()
significant_vars = significant_vars[['variable', 'coefficient', 'p_value', 'sig']]
# Create DataFrame for summary
dpm_result = {
    'Sector' : 'Materials',
    'Number of Observations': [num_obs],
    'Number of Groups': [num_groups],
    'Number of Instruments': '42',
    'Hansen Test p-value': [hansen_p_value],
    'Arellano-Bond AR(1) p-value': [ar1_p_value],
    'Arellano-Bond AR(2) p-value': [ar2_p_value],
}

# Add coefficients and p-values for significant variables
for index, row in significant_vars.iterrows():
    p_value_with_sig = f"{row['p_value']:.3e} ({row['sig']})" if pd.notna(row['sig']) else f"{row['p_value']:.3e}"
    dpm_result[f'Coef_{row["variable"]}'] = row['coefficient']
    dpm_result[f'P-Value_{row["variable"]}'] = p_value_with_sig

results_dpm_Materials = pd.DataFrame(dpm_result)

'''
result_df_sector = pd.concat([results_dpm_CommService, results_dpm_ConsumerDiscr, 
                              results_dpm_ConsuStaple, results_dpm_Energy, results_dpm_Financials, 
                              results_dpm_Industrials, results_dpm_IT, results_dpm_Materials, 
                              results_dpm_RealEstate, results_dpm_Utilities, results_dpm_Health], axis=0)
 
'''

#%% H2: Observing Time Lags

# Cluster 1
df = df_cluster0.reset_index()
command_str='lnIdioReturn L(1:7).lnIdioReturn EnvironmentalScore L(1:3).EnvironmentalScore SocialScore L(1:3).SocialScore GovernanceScore L(1:3).GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:7) iv(lnMarketCap) | collapse'
mydpd = regression.abond(command_str, df, ['Identifier', 'Year'])

def extract_results_DPM():
    m = mydpd.models[0]
    num_obs = m.num_obs
    num_groups = m.N
    hansen_p_value = m.hansen.p_value
    ar1_p_value = m.AR_list[0].P_value
    ar2_p_value = m.AR_list[1].P_value
    
    # Extracting coefficients and p-values
    regression_table = m.regression_table
    significant_vars = regression_table[regression_table['sig'] != ''].copy()
    significant_vars = significant_vars[['variable', 'coefficient', 'p_value', 'sig']]
    # Create DataFrame for summary
    dpm_result = {
        'Cluster' : '1',
        'Number of Observations': [num_obs],
        'Number of Groups': [num_groups],
        'Number of Instruments': '30',
        'Hansen Test p-value': [hansen_p_value],
        'Arellano-Bond AR(1) p-value': [ar1_p_value],
        'Arellano-Bond AR(2) p-value': [ar2_p_value],
    }
    
    # Add coefficients and p-values for significant variables
    for index, row in significant_vars.iterrows():
        p_value_with_sig = f"{row['p_value']:.3e} ({row['sig']})" if pd.notna(row['sig']) else f"{row['p_value']:.3e}"
        dpm_result[f'Coef_{row["variable"]}'] = row['coefficient']
        dpm_result[f'P-Value_{row["variable"]}'] = p_value_with_sig
    
    results_dpm_1 = pd.DataFrame(dpm_result)

    return results_dpm_1
results_dpm_1 = extract_results_DPM()


# Cluster 2
df = df_cluster1.reset_index()
command_str='lnIdioReturn L(1:3).lnIdioReturn EnvironmentalScore L(1:3).EnvironmentalScore SocialScore L(1:3).SocialScore GovernanceScore L(1:3).GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:10) iv(lnMarketCap) | collapse'
mydpd = regression.abond(command_str, df, ['Identifier', 'Year'])

def extract_results_DPM():
    m = mydpd.models[0]
    num_obs = m.num_obs
    num_groups = m.N
    hansen_p_value = m.hansen.p_value
    ar1_p_value = m.AR_list[0].P_value
    ar2_p_value = m.AR_list[1].P_value
    
    # Extracting coefficients and p-values
    regression_table = m.regression_table
    significant_vars = regression_table[regression_table['sig'] != ''].copy()
    significant_vars = significant_vars[['variable', 'coefficient', 'p_value', 'sig']]
    # Create DataFrame for summary
    dpm_result = {
        'Cluster' : '2',
        'Number of Observations': [num_obs],
        'Number of Groups': [num_groups],
        'Number of Instruments': '42',
        'Hansen Test p-value': [hansen_p_value],
        'Arellano-Bond AR(1) p-value': [ar1_p_value],
        'Arellano-Bond AR(2) p-value': [ar2_p_value],
    }
    
    # Add coefficients and p-values for significant variables
    for index, row in significant_vars.iterrows():
        p_value_with_sig = f"{row['p_value']:.3e} ({row['sig']})" if pd.notna(row['sig']) else f"{row['p_value']:.3e}"
        dpm_result[f'Coef_{row["variable"]}'] = row['coefficient']
        dpm_result[f'P-Value_{row["variable"]}'] = p_value_with_sig
    
    results_dpm_2 = pd.DataFrame(dpm_result)

    return results_dpm_2
results_dpm_2 = extract_results_DPM()

# Cluster 3

df = df_cluster2.reset_index()
command_str='lnIdioReturn L(1:3).lnIdioReturn EnvironmentalScore L(1:3).EnvironmentalScore SocialScore L(1:3).SocialScore GovernanceScore L(1:3).GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:7) iv(lnMarketCap) | collapse'
mydpd = regression.abond(command_str, df, ['Identifier', 'Year'])

def extract_results_DPM():
    m = mydpd.models[0]
    num_obs = m.num_obs
    num_groups = m.N
    hansen_p_value = m.hansen.p_value
    ar1_p_value = m.AR_list[0].P_value
    ar2_p_value = m.AR_list[1].P_value
    
    # Extracting coefficients and p-values
    regression_table = m.regression_table
    significant_vars = regression_table[regression_table['sig'] != ''].copy()
    significant_vars = significant_vars[['variable', 'coefficient', 'p_value', 'sig']]
    # Create DataFrame for summary
    dpm_result = {
        'Cluster' : '3',
        'Number of Observations': [num_obs],
        'Number of Groups': [num_groups],
        'Number of Instruments': '30',
        'Hansen Test p-value': [hansen_p_value],
        'Arellano-Bond AR(1) p-value': [ar1_p_value],
        'Arellano-Bond AR(2) p-value': [ar2_p_value],
    }
    
    # Add coefficients and p-values for significant variables
    for index, row in significant_vars.iterrows():
        p_value_with_sig = f"{row['p_value']:.3e} ({row['sig']})" if pd.notna(row['sig']) else f"{row['p_value']:.3e}"
        dpm_result[f'Coef_{row["variable"]}'] = row['coefficient']
        dpm_result[f'P-Value_{row["variable"]}'] = p_value_with_sig
    
    results_dpm_3 = pd.DataFrame(dpm_result)

    return results_dpm_3
results_dpm_3 = extract_results_DPM()

# Cluster 4

df = df_cluster3.reset_index()
command_str='lnIdioReturn L(1:5).lnIdioReturn EnvironmentalScore L(1:2).EnvironmentalScore SocialScore L(1:2).SocialScore GovernanceScore L(1:2).GovernanceScore | gmm(lnIdioReturn EnvironmentalScore SocialScore GovernanceScore, 2:6) iv(lnMarketCap) | collapse'
mydpd = regression.abond(command_str, df, ['Identifier', 'Year'])

def extract_results_DPM():
    m = mydpd.models[0]
    num_obs = m.num_obs
    num_groups = m.N
    hansen_p_value = m.hansen.p_value
    ar1_p_value = m.AR_list[0].P_value
    ar2_p_value = m.AR_list[1].P_value
    
    # Extracting coefficients and p-values
    regression_table = m.regression_table
    significant_vars = regression_table[regression_table['sig'] != ''].copy()
    significant_vars = significant_vars[['variable', 'coefficient', 'p_value', 'sig']]
    # Create DataFrame for summary
    dpm_result = {
        'Cluster' : '4',
        'Number of Observations': [num_obs],
        'Number of Groups': [num_groups],
        'Number of Instruments': '26',
        'Hansen Test p-value': [hansen_p_value],
        'Arellano-Bond AR(1) p-value': [ar1_p_value],
        'Arellano-Bond AR(2) p-value': [ar2_p_value],
    }
    
    # Add coefficients and p-values for significant variables
    for index, row in significant_vars.iterrows():
        p_value_with_sig = f"{row['p_value']:.3e} ({row['sig']})" if pd.notna(row['sig']) else f"{row['p_value']:.3e}"
        dpm_result[f'Coef_{row["variable"]}'] = row['coefficient']
        dpm_result[f'P-Value_{row["variable"]}'] = p_value_with_sig
    
    results_dpm_4 = pd.DataFrame(dpm_result)

    return results_dpm_4
results_dpm_4 = extract_results_DPM()


result_df_H2 = pd.concat([results_dpm_1 , results_dpm_2, results_dpm_3, results_dpm_4], axis=0)




















