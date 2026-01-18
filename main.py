#!/usr/bin/env python3
"""
Main driver behavior analysis pipeline
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_processor import DataProcessor
from src.feature_extractor import FeatureExtractor
from src.clustering import DriverClustering
from src.utils import save_results, create_report

def main(config_path: str = "config/config.yaml"):
    """Main analysis pipeline"""
    
    print("=" * 60)
    print("DRIVER BEHAVIOR & RISK ANALYSIS SYSTEM")
    print("=" * 60)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create results directory
    results_dir = Path(config['data']['output_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Data Processing
    print("\n[1/5] Processing raw data...")
    processor = DataProcessor(config_path)
    raw_data = processor.load_data()
    processed_data = processor.calculate_basic_metrics(raw_data)
    
    # Step 2: Feature Extraction
    print("\n[2/5] Extracting features...")
    extractor = FeatureExtractor()
    features_data = extractor.extract_time_based_features(processed_data)
    features_data = extractor.extract_acceleration_features(features_data)
    features_data = extractor.extract_speed_features(features_data)
    features_data = extractor.extract_rpm_features(features_data)
    features_data = extractor.extract_composite_features(features_data)
    
    # Step 3: Risk Scoring
    print("\n[3/5] Calculating risk scores...")
    risk_data = processor.calculate_risk_scores(features_data)
    
    # Step 4: Clustering Analysis
    print("\n[4/5] Performing clustering analysis...")
    clustering_data = processor.prepare_clustering_data(risk_data)
    
    cluster_analyzer = DriverClustering(config_path)
    optimal_k = cluster_analyzer.find_optimal_clusters(clustering_data)
    
    print(f"Optimal number of clusters: {optimal_k}")
    
    clustered_data, kmeans_model = cluster_analyzer.perform_clustering(
        clustering_data, 
        n_clusters=optimal_k
    )
    
    # Add clusters to risk data
    risk_data['cluster'] = clustered_data['cluster']
    
    # Analyze clusters
    cluster_summary = cluster_analyzer.analyze_clusters(risk_data)
    print("\nCluster Summary:")
    print(cluster_summary)
    
    # Step 5: Visualizations
    print("\n[5/5] Creating visualizations...")
    cluster_analyzer.visualize_clusters(risk_data)
    
    # Save results
    print("\nSaving results...")
    
    # Save processed data
    processor.save_processed_data(risk_data, "driver_behavior_processed.csv")
    
    # Save cluster assignments
    cluster_assignments = risk_data[['driver_id', 'cluster', 'risk_category', 
                                   'overall_risk_score', 'driver_style']]
    cluster_assignments.to_csv(results_dir / "cluster_assignments.csv", index=False)
    
    # Save cluster summary
    cluster_summary.to_csv(results_dir / "cluster_summary.csv")
    
    # Save for Power BI
    powerbi_data = risk_data[[
        'driver_id', 'cluster', 'risk_category', 'overall_risk_score',
        'driver_style', 'safety_score', 'fuel_efficiency_composite',
        'aggressive_index', 'harsh_accel_count', 'harsh_brake_count',
        'speed_mean', 'speed_p90', 'rpm_efficiency_score'
    ]]
    powerbi_data.to_csv("powerbi/driver_analysis_dataset.csv", index=False)
    
    # Generate report
    report = create_report(risk_data, cluster_summary, config)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nKey Findings:")
    print(f"- Total drivers analyzed: {len(risk_data)}")
    print(f"- Risk categories identified: {risk_data['risk_category'].unique().tolist()}")
    print(f"- Clusters found: {optimal_k}")
    print(f"- Average risk score: {risk_data['overall_risk_score'].mean():.1f}")
    print(f"\nResults saved to: {results_dir}")
    print(f"Power BI dataset: powerbi/driver_analysis_dataset.csv")
    
    return risk_data, cluster_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver Behavior Analysis System")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()
    
    main(args.config)