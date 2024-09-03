# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:26:59 2024

@author: bianc
# H1: Quantile Regression to undersatand trend overtime of ESG scores
"""

#%% Quantile Regression

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
from sklearn.utils.fixes import parse_version, sp_version # This is line is to avoid incompatibility if older SciPy version.
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point" # You should use `solver="highs"` with recent version of SciPy.
from  pydynpd import regression
import seaborn as sns

from DataProcessing import clean_data
from ClusteringModule import cluster_dataframe

ESG_dataset, df_ESG, df_numeric_std, df_ESG_std = clean_data()
df_clust_std, df_cluster0, df_cluster1, df_cluster2, df_cluster3 = cluster_dataframe()


sns.set(style="whitegrid", rc={'axes.titlesize': 22})
figures_dir = 'figures'

#%% Quantile Regression (25, 50 and 75%)
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html
# https://drkebede.medium.com/quantile-regression-tutorial-in-r-f2eec72c132b
# https://library.virginia.edu/data/articles/getting-started-with-quantile-regression


# Extract unique years from the MultiIndex
unique_years = df_ESG_std.index.get_level_values('Year').unique()
df = df_ESG_std.reset_index()
# Three Quantiles -------------------------------------------------------------
results_list_LGBM_ret = []

for year in df['Year'].unique():
    df_year = df[df['Year'] == year]
    
    train_df, test_df = train_test_split(df_year, test_size=0.10, shuffle=False)
    X_train = train_df[['EnvironmentalScore', 'SocialScore', 'GovernanceScore']]
    y_train = train_df['lnIdioReturn']
    X_test = test_df[['EnvironmentalScore', 'SocialScore', 'GovernanceScore']]
    y_test = test_df['lnIdioReturn']
    
    # Classifiers and predictions
    classifiers = {}
    for tau in [0.25, 0.5, 0.75]:
        clf = lgb.LGBMRegressor(objective='quantile', alpha=tau)
        clf.fit(X_train, y_train)
        preds = pd.DataFrame(clf.predict(X_test), columns=[str(tau)])
        classifiers[str(tau)] = {'clf': clf, 'predictions': preds}
    
    # Store results
    for tau in [0.25, 0.5, 0.75]:
        clf = classifiers[str(tau)]['clf']
        preds = classifiers[str(tau)]['predictions'][str(tau)]
        
        # Feature importance
        feature_importance = clf.feature_importances_
        
        # Prediction accuracy
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        
        # Pseudo R-squared
        y_pred_train = clf.predict(X_train)
        ss_res = sum((y_train - y_pred_train) ** 2)
        ss_tot = sum((y_train - y_train.mean()) ** 2)
        pseudo_r2 = 1 - (ss_res / ss_tot)
        
        results_list_LGBM_ret.append({
            'Year': year,
            'Quantile': tau,
            'Pseudo R-squared': pseudo_r2,
            'MSE': mse, #  average of the squared differences between predicted and actual values
            'RMSE': rmse, # standard deviation of the residuals
            'EnvironmentalScore Importance': feature_importance[0],
            'SocialScore Importance': feature_importance[1],
            'GovernanceScore Importance': feature_importance[2],
        })

results_LGBM_ret = pd.DataFrame(results_list_LGBM_ret)


'''
#look into changing the hyperparameters:
lgb_params = {
    'n_jobs': 1,
    'max_depth': 4,
    'min_data_in_leaf': 10,
    'subsample': 0.9,
    'n_estimators': 80,
    'learning_rate': 0.1,
    'colsample_bytree': 0.9,
    'boosting_type': 'gbdt'}
'''
#%% Median Regression 

df = df_ESG_std.reset_index()
unique_years = df['Year'].unique()

results_list_LGBM_Median = []

# Loop over each unique year
for year in unique_years:
    # Filter the DataFrame for the current year
    df_year = df[df['Year'] == year]
    # Split data into train and test sets (80% train, 20% test, maintain order)
    train_df, test_df = train_test_split(df_year, test_size=0.10, shuffle=False)
    
    # Extract features and target for training and testing
    X_train = train_df[['EnvironmentalScore', 'SocialScore', 'GovernanceScore']]
    y_train = train_df['lnIdioReturn']
    X_test = test_df[['EnvironmentalScore', 'SocialScore', 'GovernanceScore']]
    y_test = test_df['lnIdioReturn']
    
    # Initialize classifiers dictionary
    classifiers = {}
    
    # Train the model and make predictions
    for tau in [0.5]:
        clf = lgb.LGBMRegressor(objective='quantile', alpha=tau)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)  # Predictions for the test set
        
        # Feature importance
        feature_importance = clf.feature_importances_
        
        # Pseudo R-squared
        y_pred_train = clf.predict(X_train)
        ss_res = sum((y_train - y_pred_train) ** 2)
        ss_tot = sum((y_train - y_train.mean()) ** 2)
        pseudo_r2 = 1 - (ss_res / ss_tot)
        
        # Prediction accuracy
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        
        # Store the results
        results_list_LGBM_Median.append({
            'Year': year,
            'Pseudo R-squared': pseudo_r2,
            'MSE': mse,
            'EnvironmentalScore Importance': feature_importance[0],
            'SocialScore Importance': feature_importance[1],
            'GovernanceScore Importance': feature_importance[2],
        })

results_LGBM_Median = pd.DataFrame(results_list_LGBM_Median)

#%% Plot the R² and MSE ---------------------------------------------------------

fig, ax1 = plt.subplots(figsize=(12, 7))

# Plot Pseudo R-squared
sns.lineplot(data=results_LGBM_Median, x='Year', y='Pseudo R-squared', 
             color='orange', ax=ax1, label='Pseudo R-squared')
ax1.set_ylabel('Pseudo R-squared', fontsize=20)
ax1.tick_params(axis='y', labelcolor='orange', labelsize=14)
ax1.set_xlabel('Year', fontsize=20)

# Create a secondary y-axis for MSE
ax2 = ax1.twinx()
sns.lineplot(data=results_LGBM_Median, x='Year', y='MSE', 
             color='DarkTurquoise', ax=ax2, label='MSE')
ax2.set_ylabel('MSE',  fontsize=20)
ax2.tick_params(axis='y', labelcolor='DarkTurquoise', labelsize=14)

# Add gridlines and labels
ax1.set_xlim(results_LGBM_Median['Year'].min(), 
             results_LGBM_Median['Year'].max())
ax1.legend(loc='center', bbox_to_anchor=(0.5, 0.93), fontsize=16)
ax2.legend(loc='center', bbox_to_anchor=(0.5, 0.84), fontsize=16)
ax1.grid(True, linewidth=0.7)
ax2.grid(False)
ax1.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='x', labelsize=16)

plt.savefig(os.path.join(figures_dir, 'QuantileReg_Eval.pdf'), bbox_inches='tight')

# Plot the Variabel Importance -------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 7))
sns.lineplot(data=results_LGBM_Median, x='Year', y='EnvironmentalScore Importance', 
             color='yellowgreen', ax=ax, label='Environmental Variable Importance')
sns.lineplot(data=results_LGBM_Median, x='Year', y='SocialScore Importance', 
             color='orchid', ax=ax, label='Social Variable Importance')
sns.lineplot(data=results_LGBM_Median, x='Year', y='GovernanceScore Importance', 
             color='LightSkyBlue', ax=ax, label='Governance Variable Importance')
ax.set_ylabel('Feature Importance', fontsize=20)
ax.set_xlabel('Year', fontsize=20)
ax.set_xlim(results_LGBM_Median['Year'].min(), 
             results_LGBM_Median['Year'].max())
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.grid(True, linewidth=0.7)
ax.legend(loc='best',  fontsize=16)

plt.savefig(os.path.join(figures_dir, 'QuantileReg_FeatureImport.pdf'), bbox_inches='tight')






