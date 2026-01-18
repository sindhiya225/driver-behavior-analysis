import pandas as pd
import numpy as np
from typing import Tuple, Dict
import yaml
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Process and clean driver behavior data"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def load_data(self) -> pd.DataFrame:
        """Load and initial processing of telematics data"""
        df = pd.read_csv(self.config['data']['input_file'])
        print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def calculate_basic_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics from raw data"""
        
        # Create driver IDs if not present
        if 'driver_id' not in df.columns:
            df['driver_id'] = range(1, len(df) + 1)
        
        # Speed metrics
        speed_columns = [col for col in df.columns if 'speed' in col and '_std' not in col]
        df['avg_speed'] = df[speed_columns].mean(axis=1)
        df['max_speed'] = df[speed_columns].max(axis=1)
        df['speed_variability'] = df[[col for col in df.columns if 'speed_+ve_std' in col]].mean(axis=1)
        
        # Acceleration metrics
        accel_cols = [col for col in df.columns if 'accel' in col]
        accel_positive = [col for col in accel_cols if '+ve' in col and '_std' not in col]
        accel_negative = [col for col in accel_cols if '-ve' in col and '_std' not in col]
        
        df['harsh_acceleration_count'] = (df[accel_positive] > 
                                         self.config['thresholds']['harsh_acceleration']).sum(axis=1)
        df['harsh_braking_count'] = (df[accel_negative] < 
                                    self.config['thresholds']['harsh_braking']).sum(axis=1)
        
        # RPM analysis
        rpm_cols = [col for col in df.columns if 'rpm' in col and '_std' not in col]
        df['avg_rpm'] = df[rpm_cols].mean(axis=1)
        df['rpm_variability'] = df[[col for col in df.columns if 'rpm' in col and '_std' in col]].mean(axis=1)
        
        # Time position metrics
        tpos_cols = [col for col in df.columns if 'tPos' in col and '_std' not in col]
        df['time_variability'] = df[tpos_cols].std(axis=1)
        
        return df
    
    def calculate_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive risk scores for each driver"""
        
        # Normalize metrics
        from sklearn.preprocessing import MinMaxScaler
        
        metrics_to_normalize = [
            'harsh_acceleration_count',
            'harsh_braking_count',
            'max_speed',
            'speed_variability',
            'rpm_variability'
        ]
        
        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(
            scaler.fit_transform(df[metrics_to_normalize]),
            columns=[f"{col}_norm" for col in metrics_to_normalize]
        )
        
        # Calculate individual risk components
        weights = self.config['analysis']['risk_score_weights']
        
        df['acceleration_risk'] = df_normalized['harsh_acceleration_count_norm'] * weights['acceleration']
        df['braking_risk'] = df_normalized['harsh_braking_count_norm'] * weights['braking']
        df['speeding_risk'] = (df_normalized['max_speed_norm'] + 
                              df_normalized['speed_variability_norm']) / 2 * weights['speeding']
        
        # Fuel efficiency score (inverse of risk)
        df['fuel_efficiency'] = 1 - (df['avg_rpm'] / 5000).clip(0, 1) * weights['fuel']
        
        # Overall risk score (0-100)
        df['overall_risk_score'] = (
            df['acceleration_risk'] + 
            df['braking_risk'] + 
            df['speeding_risk'] + 
            (1 - df['fuel_efficiency'])
        ) * 100
        
        # Risk categories
        conditions = [
            df['overall_risk_score'] < 30,
            df['overall_risk_score'] < 60,
            df['overall_risk_score'] < 80,
            df['overall_risk_score'] >= 80
        ]
        choices = ['Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk']
        df['risk_category'] = np.select(conditions, choices)
        
        return df
    
    def prepare_clustering_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for clustering analysis"""
        
        features = [
            'harsh_acceleration_count',
            'harsh_braking_count',
            'max_speed',
            'speed_variability',
            'avg_rpm',
            'rpm_variability',
            'time_variability',
            'fuel_efficiency'
        ]
        
        return df[features]
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_data.csv"):
        """Save processed data for further analysis"""
        output_path = f"{self.config['data']['output_dir']}/{filename}"
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
        return output_path