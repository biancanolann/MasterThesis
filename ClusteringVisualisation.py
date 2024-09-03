# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:30:56 2024

@author: bianc

This script visualises the results of the K-Prototype clusters.
"""
# %% Setup 

import pandas as pd
import numpy as np
import umap
from kmodes.kprototypes import KPrototypes
from DataProcessing import clean_data
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

ESG_dataset, df_ESG, df_numeric_std, df_ESG_std = clean_data()

sns.set(style="whitegrid", rc={'axes.titlesize': 18})
figures_dir = 'figures'

#%% K-Prototype Clustering (function)
# https://antonsruberts.github.io/kproto-audience/
df_ESG_std.info()

# Define variables
df=df_ESG_std.copy()
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





# Define the mapping of old values to new values
mapping = {0: 1, 
           1: 2, 
           2: 1, 
           3: 0}
df_clustered_std['Cluster'] = df_clustered_std['Cluster'].map(mapping)



# Region 
clust_lab = ["1", "2", "3", "4"]
plt.figure(figsize=(8, 6.5))
sns.set(style="whitegrid")
ax = sns.histplot(
    data=df_clustered_std, 
    x='Cluster', 
    hue='Region', 
    multiple='fill', 
    palette='husl',
    stat='density',  
    shrink=0.8,
    discrete=True,
    alpha=0.9)
ax.set_xlabel("Cluster")
ax.set_ylabel("Percentage of Companies")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))  # Format y-axis as percentages
ax.set_xticks(range(len(clust_lab)))
ax.set_xticklabels(clust_lab)
sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1.02, 0.8), title='Region')





def cluster_dataframe():
    #Add the cluster number to main dataset
    df_clust_std = pd.merge(df_ESG_std.reset_index(), df_clustered_std[['Cluster']], 
                            left_on='Identifier', right_index=True, how='left')
    df_clust_std.set_index(['Identifier', 'Year'], inplace=True)
    
    df_cluster0 = df_clust_std[df_clust_std['Cluster'] == 0]
    df_cluster1 = df_clust_std[df_clust_std['Cluster'] == 1]
    df_cluster2 = df_clust_std[df_clust_std['Cluster'] == 2]
    df_cluster3 = df_clust_std[df_clust_std['Cluster'] == 3]
    
    return df_clust_std, df_cluster0, df_cluster1, df_cluster2, df_cluster3

df_clust_std, df_cluster0, df_cluster1, df_cluster2, df_cluster3 = cluster_dataframe()

#%% Optimising K-Prototype parameters
#https://juandelacalle.medium.com/k-prototypes-other-statistical-techniques-to-cluster-with-categorical-and-numerical-features-a-ac809a000316

cat_cols = [8, 9, 10, 11]
num_cols =[0, 1, 2, 4, 5, 6, 7]

# Define the range of potential clusters and gamma values
clusters_range = range(4) 
gamma_range = np.linspace(0.05, 1, 10) # gamma balances the weight btw numerical and categorical features

# Placeholder variables
best_score = -1
best_clusters = None
best_gamma = None

for n_clusters in clusters_range:
    for gamma in gamma_range:
        kproto = KPrototypes(n_clusters=n_clusters, gamma=gamma, init='Huang')
        clusters = kproto.fit_predict(df_clustered_std, categorical=cat_cols)
        score = silhouette_score(df_clustered_std.iloc[:,num_cols], clusters)

        # Check if this configuration beats the best score
        if score > best_score:
            best_score = score
            best_clusters = n_clusters
            best_gamma = gamma

print(f"Best score: {best_score}")
print(f"Optimal number of clusters: {best_clusters}")
print(f"Optimal gamma value: {best_gamma}")

# Best score: 0.1399
# Optimal number of clusters: 4
# Optimal gamma value: 0.02

#%% Evaluation: Elbow method 

list_k = list(range(1, 16)) # Loop through each k value and compute SSE
sse = []
for k in list_k:
    kproto = KPrototypes(n_clusters=k, init='Cao', n_jobs=4)
    cat_cols = [8, 9, 10, 11]
    clusters = kproto.fit_predict(df_clustered_std, categorical=cat_cols)
    sse.append(kproto.cost_)

# Plot SSE against the number of clusters
plt.figure(figsize=(8, 6.5))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters $k$')
plt.ylabel('Sum of Squared Errors')
plt.title('Elbow Method for K-Prototypes Clustering')
plt.grid(True)
plt.show()
plt.savefig(os.path.join(figures_dir, 'clust_Elbow.pdf'), bbox_inches='tight')


#%% Evaluation: Silouhette Method

# Construct Dataframe
df = df_clustered_std
df['ESGscore'] = df[['EnvironmentalScore', 'GovernanceScore', 'SocialScore']].mean(axis=1)

# One-hot encode categorical columns
cat_cols = [8, 9, 10, 11]
categorical_cols = df.columns[cat_cols]
encoded_categorical = OneHotEncoder().fit_transform(df[categorical_cols]).toarray()

# Remove original categorical columns and add encoded columns
df = df.drop(categorical_cols, axis=1)
df = np.hstack([df.values, encoded_categorical])

# Get silhouette values
for k in [4]:  # Now that we have confirmed opt k = 4, only plotting for 4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.8))

    # Run the K-prototype algorithm
    kproto = KPrototypes(n_clusters=k, init='Cao', n_jobs=4)
    labels = kproto.fit_predict(df, categorical=cat_cols)
    
    # Get silhouette values
    silhouette_vals = silhouette_samples(df, labels)
    
    # Generate colors
    colors = plt.cm.viridis(np.linspace(0, 1, k))

    # Silhouette plot
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1, color=colors[i])
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)
    
    # Average silhouette score
    avg_score = np.mean(silhouette_vals)
    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    ax1.set_yticks([])
    ax1.set_xlim([-0.05, 0.34])
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster labels')
    ax1.set_title(f'Silhouette plot for k = {k}', y=1.02)
    
    x = 9
    y = 1
    # Scatter plot with "o" shaped markers
#    centroids = kproto.cluster_centroids_
    for i in np.unique(labels):
        ax2.scatter(df[labels == i, x], df[labels == i, y], color=colors[i], marker='o', facecolors='none', edgecolors=colors[i], label=f'Cluster {i + 1}')
   
    # Scatter plot for centroids with the correct x and y positions
    for i in range(k):
        ax2.scatter(centroids[i, x], centroids[i, y], marker='*', c=[colors[i]], s=250, edgecolor='black', label=f'Centroid {i + 1}')
   
    # Expand x-axis limits for better granularity
    x_min, x_max = df[:, x].min() - 0.22, df[:, x].max() + 0.22
    ax2.set_xlim(x_min, x_max)
    y_min, y_max = df[:, y].min(), 10
    ax2.set_ylim(y_min, y_max)
    
    ax2.legend(loc='upper right')
    ax2.set_xlabel('Average ESG Scores (Standardised)')
    ax2.set_ylabel('Log Idiosyncratic Total Return (Standardised)')
    ax2.set_title('K-Prototype Clustered Data', y=1.02)
    ax2.set_aspect('auto')  
    plt.tight_layout()
    plt.suptitle(f'Silhouette analysis using k = {k}', fontsize=16, fontweight='semibold', y=1.05)
plt.savefig(os.path.join(figures_dir, 'clusters_silhouette.pdf'), bbox_inches='tight')
    
#%% Visualising the Clusters

## Individual Cluster plots
fig, axs = plt.subplots(2, 2, figsize=(9, 9))
colors = plt.cm.viridis(np.linspace(0, 1, k))

for i, ax in enumerate(axs.flat):
    cluster_data = df[labels == i]  # Data points for the current cluster
    x = 9
    y = 1 
    ax.scatter(cluster_data[:, x], cluster_data[:, y], color=colors[i], marker='o', facecolors='none', edgecolors=colors[i], label=f'Cluster {i + 1}')
    ax.scatter(centroids[i, x], centroids[i, y], marker='*', color=colors[i], s=250, edgecolor='black', label='Centroid')
    
    # Set axis labels and title for each subplot
    ax.set_xlabel('Average ESG Scores (Standardised)')
    ax.set_ylabel('Log Idiosyncratic Total Return (Standardised)')
    ax.set_title(f'Cluster {i + 1}', y=1.02)
    
    # Expand axis limits for better granularity
    x_min, x_max = df[:, x].min() - 0.22, df[:, x].max() + 0.22
    y_min, y_max = df[:, y].min(), 10
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    ax.legend(loc='upper right')
    ax.set_aspect('auto')

plt.tight_layout()
#plt.suptitle('Scatter Plots for Each Cluster', fontsize=16, fontweight='semibold', y=1.02)
plt.savefig(os.path.join(figures_dir, 'clusters_indiv.pdf'), bbox_inches='tight')


## UMAP ------------------------------------------------------------------------
# Embed numerical and categorical data
fit1 = umap.UMAP(metric='l2').fit(numerical)
fit2 = umap.UMAP(metric='dice').fit(categorical)

# Augment numerical embedding with categorical data
intersection = umap.umap_.general_simplicial_set_intersection(fit1.graph_, fit2.graph_, weight=categorical_weight)
intersection = umap.umap_.reset_local_connectivity(intersection)
embedding, _ = umap.umap_.simplicial_set_embedding(fit1._raw_data, intersection, fit1.n_components,
                                                  fit1._initial_alpha, fit1._a, fit1._b,
                                                  fit1.repulsion_strength, fit1.negative_sample_rate,
                                                  200, 'random', np.random, fit1.metric,          # https://umap-learn.readthedocs.io/en/latest/api.html
                                                  fit1._metric_kwds, False, output_dens=False,    # output_dens: Determines whether the local radii of the final embedding (an inverse measure of local density) are computed and returned in addition to the embedding. If set to True, local radii of the original data are also included in the output for comparison; the output is a tuple (embedding, original local radii, embedding local radii). This option can also be used when densmap=False to calculate the densities for UMAP embeddings.
                                                  densmap_kwds={'target_n_neighbors': 15,         # densmap: specifies whether the density-augmented objective of densMAP should be used for optimization. Turning on this option generates an embedding where the local densities are encouraged to be correlated with those in the original space.
                                                                'spread': 1.0,
                                                                'min_dist': 0.1})
fig, ax = plt.subplots()
fig.set_size_inches((20, 10))
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], s=2, c=clusters, cmap='tab20b', alpha=1.0)
legend1 = ax.legend(*scatter.legend_elements(num=k),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)

#%% Extract centroid locations 

# Num centroids
numerical_cols = df_clustered_std.select_dtypes(exclude=['object', 'category']).columns
def get_num_centroids(kproto, numerical_cols):
    centroids = kproto.cluster_centroids_
    numCentroids_df = pd.DataFrame(centroids[:, :len(numerical_cols)], columns=numerical_cols)
    numCentroids_df['Cluster'] = range(len(numCentroids_df))
    numCentroids_df = numCentroids_df.set_index('Cluster')
    numCentroids_df = numCentroids_df.apply(pd.to_numeric)
    
    return numCentroids_df

numCentroids_df = get_num_centroids(kproto, numerical_cols)

# Plot heatmap
clust_lab = ["1", "2", "3", "4"]
var_lab = ["Log Market \nCapitalisation", 
           "Log Idiosyncratic \n Total Returns", 
           "Environmental \nScore", 
           "Governance \nScore", 
           "Social\n Score", 
           "Change in \n Environmental Score", 
           "Change in \n Governance Score", 
           "Change in \n Social Score", 
           "Average \nESG score"]

plt.figure(figsize=(12, 7))
ax = sns.heatmap(numCentroids_df, annot=True, fmt=".3f", cmap="magma", 
                 annot_kws={"size": 12, "weight": "bold"},  
                 cbar_kws={"shrink": 0.8, "label": "Color Scale"},  
                 linewidths=0.5,  
                 linecolor='black')  
ax.set_yticklabels(clust_lab, fontsize=12)
ax.set_xticklabels(var_lab, fontsize=12) 
plt.ylabel('Clusters', fontsize=14)
plt.savefig(os.path.join(figures_dir, 'clust_heatmap.pdf'), bbox_inches='tight')

# %%Clustering Visualisation  - Categorical vars
clust_lab = ["1", "2", "3", "4"]

# Sector 
plt.figure(figsize=(8, 6.5))
sns.set(style="whitegrid")
ax = sns.histplot(
    data=df_clustered_std, 
    x='Cluster', 
    hue='sectorName', 
    multiple='fill', 
    palette='husl',
    stat='density',  
    shrink=0.8,
    discrete=True,
    alpha=0.9)
ax.set_xlabel("Cluster")
ax.set_ylabel("Percentage of Companies")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))  # Format y-axis as percentages
ax.set_xticks(range(len(clust_lab)))
ax.set_xticklabels(clust_lab)
sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1.02, 0.8), title='Sector Name')
plt.savefig(os.path.join(figures_dir, 'clust_Sector.pdf'), bbox_inches='tight')

# Region 
plt.figure(figsize=(8, 6.5))
sns.set(style="whitegrid")
ax = sns.histplot(
    data=df_clustered_std, 
    x='Cluster', 
    hue='Region', 
    multiple='fill', 
    palette='husl',
    stat='density',  
    shrink=0.8,
    discrete=True,
    alpha=0.9)
ax.set_xlabel("Cluster")
ax.set_ylabel("Percentage of Companies")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))  # Format y-axis as percentages
ax.set_xticks(range(len(clust_lab)))
ax.set_xticklabels(clust_lab)
sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1.02, 0.8), title='Region')
plt.savefig(os.path.join(figures_dir, 'clust_Region.pdf'), bbox_inches='tight')


# Developed 
colours =  {'Developed': 'Plum',
            'Emerging': 'CornflowerBlue'}

plt.figure(figsize=(8, 6.5))
sns.set(style="whitegrid")
ax = sns.histplot(
    data=df_clustered_std, 
    x='Cluster', 
    hue='Development', 
    multiple='fill', 
    palette=colours,
    stat='proportion',  
    shrink=0.8,
    discrete=True,
    alpha=0.9)
ax.set_xlabel("Cluster")
ax.set_ylabel("Percentage of Companies")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))  # Format y-axis as percentages
ax.set_xticks(range(len(clust_lab)))
ax.set_xticklabels(clust_lab)
sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1.02, 0.8), title='Development')
plt.savefig(os.path.join(figures_dir, 'clust_Dev.pdf'), bbox_inches='tight')








