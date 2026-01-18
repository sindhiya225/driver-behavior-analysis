import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import yaml

class DriverClustering:
    """Perform clustering analysis on driver behavior"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def find_optimal_clusters(self, data: pd.DataFrame, max_k: int = 10) -> int:
        """Determine optimal number of clusters using elbow method and silhouette score"""
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        wcss = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.config['clustering']['random_state'])
            kmeans.fit(scaled_data)
            wcss.append(kmeans.inertia_)
            
            if k > 1:
                silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
        
        # Find elbow point
        from kneed import KneeLocator
        kl = KneeLocator(list(k_range), wcss, curve='convex', direction='decreasing')
        optimal_k = kl.elbow
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow method
        ax1.plot(k_range, wcss, 'bo-')
        ax1.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('WCSS')
        ax1.set_title('Elbow Method')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette score
        ax2.plot(list(k_range)[1:], silhouette_scores, 'go-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/clusters/optimal_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return optimal_k
    
    def perform_clustering(self, data: pd.DataFrame, n_clusters: int = None) -> Tuple[pd.DataFrame, KMeans]:
        """Perform K-means clustering on driver data"""
        
        if n_clusters is None:
            n_clusters = self.config['clustering']['n_clusters']
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Apply K-means
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.config['clustering']['random_state'],
            n_init=10
        )
        
        clusters = kmeans.fit_predict(scaled_data)
        
        # Calculate cluster statistics
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = clusters
        data_with_clusters['cluster_label'] = data_with_clusters['cluster'].apply(
            lambda x: f'Cluster {x+1}'
        )
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(scaled_data, clusters)
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        # Davies-Bouldin Index (lower is better)
        db_index = davies_bouldin_score(scaled_data, clusters)
        print(f"Davies-Bouldin Index: {db_index:.3f}")
        
        return data_with_clusters, kmeans
    
    def analyze_clusters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze characteristics of each cluster"""
        
        cluster_summary = data.groupby('cluster').agg({
            'harsh_acceleration_count': 'mean',
            'harsh_braking_count': 'mean',
            'max_speed': 'mean',
            'fuel_efficiency': 'mean',
            'overall_risk_score': 'mean'
        }).round(2)
        
        cluster_summary['driver_count'] = data.groupby('cluster').size()
        cluster_summary['percentage'] = (cluster_summary['driver_count'] / len(data) * 100).round(1)
        
        # Assign cluster labels based on behavior
        cluster_labels = []
        for idx, row in cluster_summary.iterrows():
            if row['overall_risk_score'] > 70:
                label = "Aggressive Risky Drivers"
            elif row['fuel_efficiency'] < 0.4:
                label = "Fuel Inefficient Drivers"
            elif row['harsh_braking_count'] > row['harsh_acceleration_count']:
                label = "Defensive but Harsh Brakers"
            elif row['max_speed'] > 85:
                label = "Speeding Drivers"
            else:
                label = "Safe & Efficient Drivers"
            
            cluster_labels.append(label)
        
        cluster_summary['cluster_label'] = cluster_labels
        
        return cluster_summary
    
    def visualize_clusters(self, data: pd.DataFrame, features: List[str] = None):
        """Create visualizations of clusters"""
        
        if features is None:
            features = ['harsh_acceleration_count', 'harsh_braking_count', 
                       'max_speed', 'fuel_efficiency']
        
        # PCA for 2D visualization
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[features])
        
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_data)
        
        pca_df = pd.DataFrame(data=principal_components, 
                             columns=['PC1', 'PC2'])
        pca_df['cluster'] = data['cluster']
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. PCA Scatter plot
        scatter = axes[0, 0].scatter(pca_df['PC1'], pca_df['PC2'], 
                                    c=pca_df['cluster'], cmap='viridis', alpha=0.6)
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0, 0].set_title('Driver Clusters (PCA Reduced)')
        axes[0, 0].legend(*scatter.legend_elements(), title="Clusters")
        
        # 2. Parallel coordinates plot
        from pandas.plotting import parallel_coordinates
        parallel_data = data[features + ['cluster']].copy()
        parallel_data['cluster'] = parallel_data['cluster'].astype(str)
        
        parallel_coordinates(parallel_data, 'cluster', ax=axes[0, 1], alpha=0.5)
        axes[0, 1].set_title('Parallel Coordinates Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cluster distribution
        cluster_counts = data['cluster'].value_counts().sort_index()
        axes[1, 0].bar(cluster_counts.index.astype(str), cluster_counts.values)
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Number of Drivers')
        axes[1, 0].set_title('Cluster Distribution')
        
        # 4. Feature importance in PCA
        pca_loadings = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2'],
            index=features
        )
        
        pca_loadings.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('PCA Feature Loadings')
        axes[1, 1].set_ylabel('Loading Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/clusters/cluster_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()