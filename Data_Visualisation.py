# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:28:03 2024

This script provides:
    - exploratory analysis (pairs plot, correlation heatmap)
    - visualisation of the time series data (individual and aggregate)
    - plots the progression/trajectory of companie's ESG scores (clustering per region/sector)

@author: bianc
"""
# %% Setup 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
import os

from DataProcessing import clean_data
ESG_dataset, df_ESG, df_numeric_std, df_ESG_std = clean_data()

sns.set(style="whitegrid", rc={'axes.titlesize': 18})
figures_dir = 'figures'

#%% Exploratory Analysis

sns.pairplot(data=df_numeric_std)

# By Region 
sns.pairplot(data=df_ESG, hue = 'Region',
             plot_kws={'s': 6, 'alpha': 0.4})

# By Sector 
df_sectionBySectors = df_ESG[df_ESG['sectorName'].isin(df_ESG['sectorName'].unique()[8:11])]
sns.pairplot(data=df_sectionBySectors, hue = 'sectorName',
             plot_kws={'s': 6, 'alpha': 0.4})

# By Year
g = sns.pairplot(data=df_ESG, hue = 'Year', palette='viridis_r',
             plot_kws={'s': 6, 'alpha': 0.4})

descript_stats_yearly = ESG_dataset.groupby('Year').agg({
    'return': ['mean', 'std'],
    'marketCap': ['mean', 'std'],
    'EnvironmentalScore': ['mean', 'std'],
    'SocialScore': ['mean', 'std'],
    'GovernanceScore': ['mean', 'std'] })

descript_stats_sector = ESG_dataset.groupby('sectorName').agg({
    'return': ['mean', 'std'],
    'marketCap': ['mean', 'std'],
    'EnvironmentalScore': ['mean', 'std'],
    'SocialScore': ['mean', 'std'],
    'GovernanceScore': ['mean', 'std'] })

descript_stats_region = ESG_dataset.groupby('Region').agg({
    'return': ['mean', 'std'],
    'marketCap': ['mean', 'std'],
    'EnvironmentalScore': ['mean', 'std'],
    'SocialScore': ['mean', 'std'],
    'GovernanceScore': ['mean', 'std'] })

# Correlation matrix of features 
corr_mat = df_numeric_std.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_mat, annot=True, cmap='viridis_r', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numeric Variables')
plt.show()

#%% Data distribution

# Year timeframes per company
year_count = df_ESG_std.groupby(level=0).apply(lambda x: x.index.get_level_values(level=1).nunique()).value_counts().sort_index(ascending=False).reset_index()
year_count.info()
year_count.columns = ['Years', 'Companies']
# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=year_count, x='Years', y='Companies', color='turquoise')
#plt.title('Number of Years of Data Available for Each Company', fontsize=18)
plt.xlabel('Number of years of observed data', fontsize='16')
plt.ylabel('Number of companies', fontsize='16')
plt.savefig(os.path.join(figures_dir, 'year_count.pdf'), bbox_inches='tight')

# Sectors distribution
plt.figure(figsize=(14, 7))
husl_palette = sns.color_palette("husl", len(df_ESG_std['sectorName'].value_counts())) 
sector_counts = df_ESG_std['sectorName'].value_counts() 
sector_percents = sector_counts / sector_counts.sum() * 100 
plt.pie(sector_counts, labels=sector_counts.index, autopct=None, 
        startangle=140, textprops={'fontsize': 14}, colors=husl_palette) 
plt.legend(
    labels=[f'{name} ({pct:.1f}%)' for name, pct in zip(sector_counts.index, sector_percents)],
    loc='center left',
    bbox_to_anchor=(1.2, 0.5),
    title='Sector Name',
    title_fontsize='20',
    fontsize='16')
#plt.title('Distribution of Company Sectors', fontsize=20, pad=18)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'sector_distr.pdf'), bbox_inches='tight')

# Region distribution
#fig 1, axes = plt.subplots(1, 1, figsize=(12, 6))
plt.figure(figsize=(12, 7))
husl_palette = sns.color_palette("husl", len(df_ESG_std['Region'].value_counts())) 
sector_counts = df_ESG_std['Region'].value_counts() 
sector_percents = sector_counts / sector_counts.sum() * 100 
plt.pie(sector_counts, labels=sector_counts.index, autopct=None, 
        textprops={'fontsize': 14}, startangle=140, colors=husl_palette) 
plt.legend(
    labels=[f'{name} ({pct:.1f}%)' for name, pct in zip(sector_counts.index, sector_percents)],
    loc='center left',
    bbox_to_anchor=(1.2, 0.5),
    title='Region',
    title_fontsize='20',
    fontsize='14')
#plt.title('Distribution of Company Regions', fontsize=20, pad=18)
plt.savefig(os.path.join(figures_dir, 'region_distr.pdf'), bbox_inches='tight')


#%% ESG score count
bins = range(0, 12)  
def plot_and_save_pie_for_score(score_column, title, filename):
    # create the bins
    binned_scores = pd.cut(score_column, bins=bins, right=False)
    bin_counts = binned_scores.value_counts().sort_index()
    bin_percents = bin_counts / bin_counts.sum() * 100
    bin_labels = [str(int(interval.left)) for interval in bin_counts.index]
    # plot pie chart
    husl_palette = sns.color_palette("viridis_r", len(bin_counts)) 
    plt.figure(figsize=(12, 7))
    plt.pie(bin_counts, labels=bin_labels, autopct=None, counterclock=False, startangle=90, colors=husl_palette)
    plt.legend(
        labels=[f'{label} ({pct:.1f}%)' for label, pct in zip(bin_labels, bin_percents)],
        loc='center left',
        bbox_to_anchor=(1.2, 0.5),
        title='Score Range',
        title_fontsize='13',
        fontsize='12' )
    plt.title(title, fontsize=20, pad=18)
    plt.savefig(os.path.join(figures_dir, filename), bbox_inches='tight')

# Plot and save pie charts for EnvironmentalScore, SocialScore, and GovernanceScore
plot_and_save_pie_for_score(ESG_dataset['EnvironmentalScore'], 
                            'Distribution of Environmental Scores', 'distr_Envir.pdf')
plot_and_save_pie_for_score(ESG_dataset['SocialScore'], 'Distribution of Social Scores', 'distr_Soc.pdf')
plot_and_save_pie_for_score(ESG_dataset['GovernanceScore'], 'Distribution of Governance Scores', 'distr_Gov.pdf')

#---------------------------------------------------------------------------------------

# Generate the binned scores
bins = range(0, 12)
labels = [f'{i}' for i in range(len(bins) - 1)]

def get_binned_counts(score_column):
    binned_scores = pd.cut(score_column, bins=bins, right=False)
    bin_counts = binned_scores.value_counts().sort_index()
    return bin_counts
binned_scores = {
    'Environmental Score': get_binned_counts(ESG_dataset['EnvironmentalScore']),
    'Social Score': get_binned_counts(ESG_dataset['SocialScore']),
    'Governance Score': get_binned_counts(ESG_dataset['GovernanceScore']) }
bin_labels = [str(int(interval.left)) for interval in binned_scores['Environmental Score'].index]

def plot_esg_scores(binned_scores, bin_labels):
    labels = list(binned_scores.keys())
    data= np.array(list(binned_scores.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    
    for i, (colname, color) in enumerate(zip(bin_labels, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)
    r, g, b, _ = color
    ax.bar_label(rects, label_type='center', color='white' )
    ax.legend(ncols=len(bin_labels), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax

plot_esg_scores(binned_scores, bin_labels)




#%% Exploratory Data Distribution and Data Transformation

## Part 2: Variable Distributions Analysis Pre-transformations
# Descriptive statistics : overall
descript_stats = pd.DataFrame(ESG_dataset[['IdioReturn', 'marketCap', 'EnvironmentalScore', 'SocialScore', 'GovernanceScore']].describe()).drop('count').T
descript_stats['skewness'] = ESG_dataset[['IdioReturn', 'marketCap', 'EnvironmentalScore', 'SocialScore', 'GovernanceScore']].skew()
descript_stats['kurtosis'] = ESG_dataset[['IdioReturn', 'marketCap', 'EnvironmentalScore', 'SocialScore', 'GovernanceScore']].kurtosis()
descript_stats = descript_stats.T

# Raw overall var distr
fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
var_list = ["marketCap", 'IdioReturn', 'ESGscore', 'EnvironmentalScore', 'SocialScore', 'GovernanceScore']
titles = ['Market Capitalisation', 'Idiosyncratic Total Return', 'ESG Score', 'Environmental Score', 'Social Score', 'Governance Score']
x_labels = ['Market Capitalisation (million US$)', 'Idiosyncratic Total Return (%)', 'Average ESG Score', 'Environmental Score', 'Social Score', 'Governance Score']
for ax, var, title, xlabel in zip(axes.flat, var_list, titles, x_labels):
    sns.histplot(ESG_dataset[var], kde=True, ax=ax, color='blue', bins=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
plt.subplots_adjust(hspace=0.32)
#plt.suptitle('Numeric Variable Distribution', fontsize=22)
plt.savefig(os.path.join(figures_dir, 'Raw_allVar.pdf'), bbox_inches='tight')





## Part 3: Transformation
# Log Transformations of Mkt Cap and Total Return
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
var_list = ['IdioReturn','lnIdioReturn','marketCap', 'lnMarketCap']
titles = ['Idiosyncratic Total Return','Log Idiosyncratic Total Return','Market Capitalisation', 'Log Market Capitalisation']
x_labels = ['Idiosyncratic Total Return (%)', 'Idiosyncratic Total Return (%)', 'Market Capitalisation (million US$)', 'Market Capitalisation (million US$)']
for ax, var, title, xlabel in zip(axes.flat, var_list, titles, x_labels):
    sns.histplot(ESG_dataset[var], kde=True, ax=ax, color='blue', bins=45)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
plt.subplots_adjust(hspace=0.32)
#plt.suptitle('Comparing the Distributions of Idiosyncratic Total Return and \n Market Capitalization Before and After Logarithmic Transformation', fontsize=20)
plt.savefig(os.path.join(figures_dir, 'log_CFP.pdf'), bbox_inches='tight')

## Part 4: Scaling
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler, RobustScaler, MinMaxScaler
df_ESG.info()
numeric_col_indices = [4, 5, 6, 7, 8, 9] 
df_numeric_std = df_ESG.iloc[:,numeric_col_indices]
#scaling = QuantileTransformer(output_distribution='normal')      
#scaling = PowerTransformer(method="yeo-johnson")
scaling = RobustScaler(quantile_range=(25, 75))
df_numeric_std = pd.DataFrame(scaling.fit_transform(df_ESG.iloc[:, numeric_col_indices]), columns=df_ESG.columns[numeric_col_indices], index=df_ESG.index)

# Transformed all var distr
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
var_list = ["lnMarketCap", 'lnIdioReturn', 'ESGscore', 'EnvironmentalScore', 'SocialScore', 'GovernanceScore']
titles = ['Log Market Capitalisation', 'Log Idiosyncratic Total Return', 'ESG score', 'Environmental Score', 'Social Score', 'Governance Score']
x_labels = ['Market Capitalisation (million US$)', 'Idiosyncratic Total Return (%)', 'Average ESG Score', 'Environmental Score', 'Social Score', 'Governance Score']
for ax, var, title, xlabel in zip(axes.flat, var_list, titles, x_labels):
    sns.histplot(df_numeric_std[var], kde=True, ax=ax, color='blue', bins=35)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
plt.subplots_adjust(hspace=0.32)
plt.savefig(os.path.join(figures_dir, 'Scaling_RobustScaler.pdf'), bbox_inches='tight')

# Post Transformation Descriptive statistics
descript_stats_transformed = pd.DataFrame(df_numeric_std[['lnIdioReturn', 'lnMarketCap','EnvironmentalScore', 'SocialScore', 'GovernanceScore']].describe()).drop('count').T
descript_stats_transformed['skewness'] = df_numeric_std[['lnIdioReturn', 'lnMarketCap','EnvironmentalScore', 'SocialScore', 'GovernanceScore']].skew()
descript_stats_transformed['kurtosis'] = df_numeric_std[['lnIdioReturn', 'lnMarketCap', 'EnvironmentalScore', 'SocialScore', 'GovernanceScore']].kurtosis()
descript_stats_transformed = descript_stats_transformed.T


df_ESG_std.info()
df_ESG_std.head()
#%% Boxplot ESG trajectory

df = ESG_dataset.copy().reset_index()
df['Year'] = df['Year'].dt.year

palette = sns.color_palette('husl', n_colors=len(df['Year']))
plt.figure(figsize=(12, 6))
sns.boxplot(x='Year', y='EnvironmentalScore', data=df)
#plt.xticks(rotation=45) 
plt.title('Boxplot of Average Yearly EnvironmentalScore')

#%% Scatterplot ESG Trajectories

# Set colours
M_colour = 'turquoise'
R_colour = 'cornflowerblue'
E_colour = 'yellowgreen'
S_colour = 'orchid' 
G_colour = 'orange'

#%% ESG progression - Dataset

ESG_trajectory = df_ESG.groupby('Identifier').agg(
    E_change=('EnvironmentalScore', lambda x: x.iloc[-1] - x.iloc[0]),
    E_last=('EnvironmentalScore', 'last'),
    G_change=('GovernanceScore', lambda x: x.iloc[-1] - x.iloc[0]),
    G_last=('GovernanceScore', 'last'),
    S_change=('SocialScore', lambda x: x.iloc[-1] - x.iloc[0]),
    S_last=('SocialScore', 'last'),
    sectorName=('sectorName', 'first'),
    Region=('Region', 'first')
).reset_index()
   
# Environmental 
Last = 'E_last'
Change = 'E_change'
colour = E_colour
ESG_score = 'Environmental'

# Social
Last = 'S_last'
Change = 'S_change'
colour = S_colour
ESG_score = 'Social'

# Governance
Last = 'G_last'
Change = 'G_change'
colour = G_colour
ESG_score = 'Governance'

#%% ESG progression - Plot 
#https://seaborn.pydata.org/generated/seaborn.jointplot.html

def trajectory_plot(data, hue, Last, Change, ESG_score, scatter_kws={'s': 8, 'alpha': 0.5}):
    # Create the jointplot without regression lines
    g = sns.jointplot(data=data, x=Last, y=Change, hue=hue, palette='husl', height=8)

    # Adjust scatter plot appearance
    for collection in g.ax_joint.collections:
        collection.set_sizes([scatter_kws['s']])
        collection.set_alpha(scatter_kws['alpha'])
    
    ax = g.ax_joint
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(-8.0, 8.0)
    
    # Create regression lines using regplot for each hue group
    unique_hues = data[hue].unique()
    colors = sns.color_palette('husl', len(unique_hues))
    
    for i, unique_value in enumerate(unique_hues):
        sns.regplot(data=data[data[hue] == unique_value],
                    x=Last, y=Change, ax=ax, scatter=False,
                    ci=85, line_kws={'color': colors[i]})
    
    # Add the legend manually
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Sector Name', loc='upper left')
    
    # Set the axis labels
    ax.set_xlabel(f'Last {ESG_score} score')
    ax.set_ylabel(f'Change in {ESG_score} score')
    
# For plotting per Sectors, set data=Trajectory_sector, change 'hue' and choose the sectors to plot in [i:n], eg:
Trajectory_sector = ESG_trajectory[ESG_trajectory['sectorName'].isin(ESG_trajectory['sectorName'].unique()[7:11])] 
trajectory_plot(data=Trajectory_sector, hue='sectorName', 
                Last='G_last', Change='G_change', ESG_score='Governance')

# For plotting per Region, set data=ESG_trajectory and change 'hue', eg:
trajectory_plot(data=ESG_trajectory, hue='Region', 
                Last='G_last', Change='G_change', ESG_score='Governance')


#%% Plot time-series data 

# Individual company ----------------------------------------------------------

def indiv_timeseries_plot (identifier):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot the market cap on the primary y-axis
    mkt = ax1.plot(df_ESG.loc[identifier, 'Year'], 
                   df_ESG.loc[identifier, 'lnMarketCap'], color=M_colour, linewidth=0.9)
    ax1.set_ylabel('Market Cap ($m)')
    
    # Create a secondary y-axis for the ESG scores
    ax2 = ax1.twinx() 
    envir = ax2.plot(df_ESG.loc[identifier, 'Year'],
             df_ESG.loc[identifier, 'EnvironmentalScore'], color=E_colour, label='EnvironmentalScore', linewidth=0.9)
    social = ax2.plot(df_ESG.loc[identifier, 'Year'],
             df_ESG.loc[identifier, 'SocialScore'], color=S_colour, label='SocialScore', linewidth=0.9)
    gov = ax2.plot(df_ESG.loc[identifier, 'Year'],
             df_ESG.loc[identifier, 'GovernanceScore'], color=G_colour, label='GovernanceScore', linewidth=0.9)
    ax2.set_ylabel('ESG Scores')
    
    # Add legend to the plot
    major_locator = AutoDateLocator() #locator that automatically chooses the best positions for major ticks on a date axis. 
    formatter = ConciseDateFormatter(major_locator) #formats date ticks on a plot's axis in a concise and readable manner.
    ax1.xaxis.set_major_formatter(formatter)
    ax2.xaxis.set_major_formatter(formatter)
    all_lines = mkt + envir + social + gov
    ax1.legend(all_lines, ['Market Cap ($m)', 'Environmental Score', 'Social Score', 'Governance Score'], loc='best')
    plt.title(f'Market Cap, ROI and ESG Scores Over Time for ID {identifier}')
    plt.grid(False)

identifier = 'MJ917817'
indiv_timeseries_plot(identifier)


# Subplot for 9 companies : MktCap --------------------------------------------

identifiers = np.random.choice(
    df_ESG.index.get_level_values('Identifier').value_counts()[lambda x: x > 1].index, 
    size=9, replace=False)

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle('Market Cap and ESG Scores Over Time for Selected Identifiers', fontsize=20)

# Loop through the selected identifiers and create subplots
for i, identifier in enumerate(identifiers):
    ax1 = axes[i // 3, i % 3]
    
    # Extract data for the current identifier
    data = df_ESG.loc[identifier]
    
    # Plot the market cap on the primary y-axis
    mkt = ax1.plot(data.index, data['lnMarketCap'], color=M_colour, linewidth=0.9, label='Log Market Cap ($m)')
    ax1.set_ylabel('Log Market Cap ($m)')
    
    # Create a secondary y-axis for the ESG scores
    ax2 = ax1.twinx()
    envir = ax2.plot(data.index, data['EnvironmentalScore'], color=E_colour, label='Environmental Score', linewidth=0.9)
    social = ax2.plot(data.index, data['SocialScore'], color=S_colour, label='Social Score', linewidth=0.9)
    gov = ax2.plot(data.index, data['GovernanceScore'], color=G_colour, label='Governance Score', linewidth=0.9)
    ax2.set_ylabel('ESG Scores')

    # Add legend to the plot
    major_locator = AutoDateLocator() #locator that automatically chooses the best positions for major ticks on a date axis. 
    formatter = ConciseDateFormatter(major_locator) #formats date ticks on a plot's axis in a concise and readable manner.
    ax1.xaxis.set_major_formatter(formatter)
    ax2.xaxis.set_major_formatter(formatter)
    all_lines = mkt + envir + social + gov
    plt.grid(False)
    
# Create a single legend for all subplots
all_lines = mkt + envir + social + gov
ax1.legend(all_lines, ['Log Market Cap ($m)', 'Environmental Score', 'Social Score', 'Governance Score'], loc='center left', bbox_to_anchor=(1.1, 0.5))
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# Subplot for 9 companies : Return --------------------------------------------

identifiers = np.random.choice(
    df_ESG.index.get_level_values('Identifier').value_counts()[lambda x: x > 1].index, 
    size=9, replace=False)

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle('ROI and ESG Scores Over Time for Selected Identifiers', fontsize=20)

# Loop through the selected identifiers and create subplots
for i, identifier in enumerate(identifiers):
    ax1 = axes[i // 3, i % 3]
    
    # Extract data for the current identifier
    data = df_ESG.loc[identifier]
    
    # Plot the market cap on the primary y-axis
    roi = ax1.plot(data.index, data['lnReturn'], color=R_colour, linewidth=0.9, label='Log ROI (%)')
    ax1.set_ylabel('Log Return on Investment (%)')
    
    # Create a secondary y-axis for the ESG scores
    ax2 = ax1.twinx()
    envir = ax2.plot(data.index, data['EnvironmentalScore'], color=E_colour, label='Environmental Score', linewidth=0.9)
    social = ax2.plot(data.index, data['SocialScore'], color=S_colour, label='Social Score', linewidth=0.9)
    gov = ax2.plot(data.index, data['GovernanceScore'], color=G_colour, label='Governance Score', linewidth=0.9)
    ax2.set_ylabel('ESG Scores')

    # Add legend to the plot
    major_locator = AutoDateLocator() #locator that automatically chooses the best positions for major ticks on a date axis. 
    formatter = ConciseDateFormatter(major_locator) #formats date ticks on a plot's axis in a concise and readable manner.
    ax1.xaxis.set_major_formatter(formatter)
    ax2.xaxis.set_major_formatter(formatter)
    all_lines = mkt + envir + social + gov
    plt.grid(False)
    
# Create a single legend for all subplots
all_lines = roi + envir + social + gov
ax1.legend(all_lines, ['Log ROI (%)', 'Environmental Score', 'Social Score', 'Governance Score'], loc='center left', bbox_to_anchor=(1.1, 0.5))
fig.tight_layout(rect=[0, 0.03, 1, 0.95])