import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List

class FeatureExtractor:
    """Extract meaningful features from raw telematics data"""
    
    def extract_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from time-position data"""
        
        # Get all tPos columns
        tpos_cols = [col for col in df.columns if 'tPos' in col]
        
        # Separate positive and negative time positions
        tpos_positive = [col for col in tpos_cols if '+ve' in col]
        tpos_negative = [col for col in tpos_cols if '-ve' in col]
        
        # Time distribution metrics
        df['time_response_mean'] = df[tpos_positive].mean(axis=1)
        df['time_response_std'] = df[tpos_positive].std(axis=1)
        df['time_reaction_mean'] = df[tpos_negative].mean(axis=1)
        df['time_reaction_std'] = df[tpos_negative].std(axis=1)
        
        # Response time percentiles
        df['time_p95'] = df[tpos_positive].quantile(0.95, axis=1)
        df['time_p99'] = df[tpos_positive].quantile(0.99, axis=1)
        
        # Calculate time consistency score
        df['time_consistency'] = 1 / (1 + df['time_response_std'])
        
        return df
    
    def extract_acceleration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from acceleration data"""
        
        accel_cols = [col for col in df.columns if 'accel' in col]
        accel_positive = [col for col in accel_cols if '+ve' in col and '_std' not in col]
        accel_negative = [col for col in accel_cols if '-ve' in col and '_std' not in col]
        
        # Aggressiveness metrics
        df['accel_aggressiveness'] = df[accel_positive].mean(axis=1)
        df['decel_aggressiveness'] = df[accel_negative].mean(axis=1)
        
        # Jerk calculation (change in acceleration)
        jerk_positive = df[accel_positive].diff(axis=1).mean(axis=1)
        jerk_negative = df[accel_negative].diff(axis=1).mean(axis=1)
        df['jerk_score'] = np.sqrt(jerk_positive**2 + jerk_negative**2)
        
        # Smooth driving score
        df['smooth_driving_score'] = 1 / (1 + df['jerk_score'])
        
        # Count of extreme events
        thresholds = {
            'very_harsh_accel': 5.0,
            'harsh_accel': 3.0,
            'very_harsh_brake': -5.0,
            'harsh_brake': -3.0
        }
        
        df['very_harsh_accel_count'] = (df[accel_positive] > thresholds['very_harsh_accel']).sum(axis=1)
        df['harsh_accel_count'] = (df[accel_positive] > thresholds['harsh_accel']).sum(axis=1)
        df['very_harsh_brake_count'] = (df[accel_negative] < thresholds['very_harsh_brake']).sum(axis=1)
        df['harsh_brake_count'] = (df[accel_negative] < thresholds['harsh_brake']).sum(axis=1)
        
        return df
    
    def extract_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from speed data"""
        
        speed_cols = [col for col in df.columns if 'speed' in col and '_std' not in col]
        
        # Speed distribution analysis
        df['speed_mean'] = df[speed_cols].mean(axis=1)
        df['speed_std'] = df[speed_cols].std(axis=1)
        df['speed_cv'] = df['speed_std'] / df['speed_mean']  # Coefficient of variation
        
        # Speeding tendency
        speed_percentiles = [0.5, 0.75, 0.9, 0.95, 0.99]
        for p in speed_percentiles:
            df[f'speed_p{p*100:.0f}'] = df[speed_cols].quantile(p, axis=1)
        
        # Speed consistency
        df['speed_consistency'] = 1 / (1 + df['speed_std'])
        
        # High speed duration estimation
        high_speed_threshold = 70  # mph
        df['high_speed_ratio'] = (df[speed_cols] > high_speed_threshold).mean(axis=1)
        
        return df
    
    def extract_rpm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from RPM data"""
        
        rpm_cols = [col for col in df.columns if 'rpm' in col and '_std' not in col]
        
        # RPM efficiency metrics
        df['rpm_mean'] = df[rpm_cols].mean(axis=1)
        df['rpm_std'] = df[rpm_cols].std(axis=1)
        df['rpm_range'] = df[rpm_cols].max(axis=1) - df[rpm_cols].min(axis=1)
        
        # RPM optimization score (lower is better for fuel efficiency)
        optimal_rpm_range = (1500, 2500)
        rpm_deviation = np.where(
            df['rpm_mean'] < optimal_rpm_range[0],
            optimal_rpm_range[0] - df['rpm_mean'],
            np.where(
                df['rpm_mean'] > optimal_rpm_range[1],
                df['rpm_mean'] - optimal_rpm_range[1],
                0
            )
        )
        df['rpm_efficiency_score'] = 1 / (1 + rpm_deviation)
        
        # RPM variability patterns
        df['rpm_variability_score'] = 1 / (1 + df['rpm_std'])
        
        return df
    
    def extract_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite features combining multiple metrics"""
        
        # Safety score
        df['safety_score'] = (
            0.3 * (1 - df['harsh_accel_count'] / df['harsh_accel_count'].max()) +
            0.3 * (1 - df['harsh_brake_count'] / df['harsh_brake_count'].max()) +
            0.2 * df['speed_consistency'] +
            0.2 * df['time_consistency']
        )
        
        # Fuel efficiency composite score
        df['fuel_efficiency_composite'] = (
            0.4 * df['rpm_efficiency_score'] +
            0.3 * df['smooth_driving_score'] +
            0.2 * (1 - df['high_speed_ratio']) +
            0.1 * df['speed_consistency']
        )
        
        # Aggressive driving index
        df['aggressive_index'] = (
            0.25 * (df['accel_aggressiveness'] / df['accel_aggressiveness'].max()) +
            0.25 * (abs(df['decel_aggressiveness']) / abs(df['decel_aggressiveness']).max()) +
            0.25 * (df['speed_p90'] / df['speed_p90'].max()) +
            0.25 * (df['jerk_score'] / df['jerk_score'].max())
        )
        
        # Driver style classification
        conditions = [
            (df['safety_score'] >= 0.8) & (df['fuel_efficiency_composite'] >= 0.7),
            (df['safety_score'] >= 0.6) & (df['fuel_efficiency_composite'] >= 0.5),
            (df['aggressive_index'] > 0.7),
            (df['fuel_efficiency_composite'] < 0.4),
            (df['safety_score'] < 0.5)
        ]
        choices = [
            'Safe & Efficient',
            'Average',
            'Aggressive',
            'Fuel Inefficient',
            'Risky'
        ]
        
        df['driver_style'] = np.select(conditions, choices, default='Unclassified')
        
        return df