# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:30:56 2024

@author: bianc

Creating the clusters using K-Prototype
"""
# %% Setup 

import pandas as pd
from kmodes.kprototypes import KPrototypes
from DataProcessing import clean_data
import seaborn as sns

ESG_dataset, df_ESG, df_numeric_std, df_ESG_std = clean_data()

#%% K-Prototype Clustering (function)
# https://antonsruberts.github.io/kproto-audience/
# Define variables
df=df_ESG_std 
k=4
categorical_weight=0.05
 # Step 1: Preprocess data
# Form dataset
grouped = df.groupby(level=0)
numeric_agg = grouped[['lnMarketCap', 'lnIdioReturn',
                       'EnvironmentalScore', 'GovernanceScore', 'SocialScore']].mean()    
numeric_agg['E_change'] = grouped['EnvironmentalScore'].apply(lambda x: x.iloc[-1] - x.iloc[0])
numeric_agg['G_change'] = grouped['GovernanceScore'].apply(lambda x: x.iloc[-1] - x.iloc[0])
numeric_agg['S_change'] = grouped['SocialScore'].apply(lambda x: x.iloc[-1] - x.iloc[0]) 
categorical_agg = grouped[['country', 'Region', 'Development', 'sectorName']].first()
df_clustered_std = numeric_agg.join(categorical_agg)

# Process numerical and categorical columns
categorical = df_clustered_std.select_dtypes(include=['category'])
categorical = pd.get_dummies(categorical)
cat_cols = [8, 9, 10, 11]
numerical = df_clustered_std.select_dtypes(exclude=['object', 'category'])

 # Step 3: K-Prototypes Clustering
kproto = KPrototypes(n_clusters=k, init='Cao', n_jobs=4) # init: method for initialisation of cluster centroids ; n_jobs: nb of parallel jobs that run the computation. 
clusters = kproto.fit_predict(df_clustered_std, categorical=cat_cols) # 1) fits of the K-Prototypes model (learn the cluster centers) and 2) predict/assigns cluster labels to each observation.
labels = kproto.labels_
centroids = kproto.cluster_centroids_
# Add cluster labels to the DataFrame
df_clustered_std['Cluster'] = labels
 

# Build a df for each seperate cluster
def cluster_dataframe():
    # Add the cluster number to main dataset
    df_clust_std = pd.merge(df_ESG_std.reset_index(), df_clustered_std[['Cluster']], 
                            left_on='Identifier', right_index=True, how='left')
    df_clust_std.set_index(['Identifier', 'Year'], inplace=True)
    
    df_cluster0 = df_clust_std[df_clust_std['Cluster'] == 0]
    df_cluster1 = df_clust_std[df_clust_std['Cluster'] == 1]
    df_cluster2 = df_clust_std[df_clust_std['Cluster'] == 2]
    df_cluster3 = df_clust_std[df_clust_std['Cluster'] == 3]
    
    return df_clust_std, df_cluster0, df_cluster1, df_cluster2, df_cluster3

df_clust_std, df_cluster0, df_cluster1, df_cluster2, df_cluster3 = cluster_dataframe()

