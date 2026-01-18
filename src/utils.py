"""
Utility functions for Driver Behavior Analysis System
"""

import pandas as pd
import numpy as np
import json
import yaml
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix

# Setup module logger
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML configuration file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        from src import DEFAULT_CONFIG
        return DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def save_config(config: Dict, config_path: str):
    """
    Save configuration to YAML file.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    config_path : str
        Path to save YAML file
    """
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        raise

def validate_data(df: pd.DataFrame, rules: Optional[Dict] = None) -> Dict:
    """
    Validate data quality and integrity.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to validate
    rules : dict, optional
        Validation rules
        
    Returns:
    --------
    dict
        Validation results
    """
    if rules is None:
        rules = {
            'min_rows': 10,
            'max_missing_percentage': 20,
            'required_columns': ['speed_+ve_0.100000', 'accel_+ve_0.100000']
        }
    
    results = {
        'passed': True,
        'checks': {},
        'warnings': [],
        'errors': []
    }
    
    # Check 1: Minimum rows
    if len(df) < rules['min_rows']:
        results['passed'] = False
        results['errors'].append(f"Too few rows: {len(df)} < {rules['min_rows']}")
    results['checks']['min_rows'] = len(df) >= rules['min_rows']
    
    # Check 2: Missing values
    missing_percentage = df.isnull().sum().max() / len(df) * 100
    if missing_percentage > rules['max_missing_percentage']:
        results['passed'] = False
        results['errors'].append(f"High missing values: {missing_percentage:.1f}%")
    results['checks']['missing_values'] = missing_percentage <= rules['max_missing_percentage']
    results['checks']['missing_percentage'] = missing_percentage
    
    # Check 3: Required columns
    missing_cols = [col for col in rules['required_columns'] if col not in df.columns]
    if missing_cols:
        results['passed'] = False
        results['errors'].append(f"Missing required columns: {missing_cols}")
    results['checks']['required_columns'] = len(missing_cols) == 0
    
    # Check 4: Data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) / len(df.columns) < 0.8:
        results['warnings'].append("Low percentage of numeric columns")
    results['checks']['numeric_columns_percentage'] = len(numeric_cols) / len(df.columns)
    
    # Check 5: Outliers (basic check)
    q1 = df[numeric_cols].quantile(0.25)
    q3 = df[numeric_cols].quantile(0.75)
    iqr = q3 - q1
    outlier_mask = ((df[numeric_cols] < (q1 - 3 * iqr)) | (df[numeric_cols] > (q3 + 3 * iqr))).any(axis=1)
    outlier_percentage = outlier_mask.sum() / len(df) * 100
    if outlier_percentage > 10:
        results['warnings'].append(f"High outlier percentage: {outlier_percentage:.1f}%")
    results['checks']['outlier_percentage'] = outlier_percentage
    
    # Check 6: Duplicates
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        results['warnings'].append(f"Duplicate rows found: {duplicate_rows}")
    results['checks']['duplicate_rows'] = duplicate_rows
    
    logger.info(f"Data validation completed: {'PASSED' if results['passed'] else 'FAILED'}")
    
    return results

def calculate_statistics(df: pd.DataFrame, columns: Optional[List] = None) -> pd.DataFrame:
    """
    Calculate comprehensive statistics for numerical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    columns : list, optional
        Columns to analyze (default: all numerical)
        
    Returns:
    --------
    pd.DataFrame
        Statistics summary
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    stats_summary = pd.DataFrame(index=columns)
    
    # Basic statistics
    stats_summary['mean'] = df[columns].mean()
    stats_summary['median'] = df[columns].median()
    stats_summary['std'] = df[columns].std()
    stats_summary['min'] = df[columns].min()
    stats_summary['max'] = df[columns].max()
    stats_summary['range'] = stats_summary['max'] - stats_summary['min']
    
    # Percentiles
    percentiles = [0.01, 0.05, 0.25, 0.75, 0.95, 0.99]
    for p in percentiles:
        stats_summary[f'p{p*100:.0f}'] = df[columns].quantile(p)
    
    # Additional statistics
    stats_summary['cv'] = stats_summary['std'] / stats_summary['mean']  # Coefficient of variation
    stats_summary['skewness'] = df[columns].apply(lambda x: stats.skew(x.dropna()))
    stats_summary['kurtosis'] = df[columns].apply(lambda x: stats.kurtosis(x.dropna()))
    
    # Missing values
    stats_summary['missing'] = df[columns].isnull().sum()
    stats_summary['missing_percentage'] = stats_summary['missing'] / len(df) * 100
    
    # Zero values
    stats_summary['zeros'] = (df[columns] == 0).sum()
    stats_summary['zero_percentage'] = stats_summary['zeros'] / len(df) * 100
    
    # Outliers (using IQR method)
    q1 = df[columns].quantile(0.25)
    q3 = df[columns].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df[columns] < (q1 - 1.5 * iqr)) | (df[columns] > (q3 + 1.5 * iqr))).sum()
    stats_summary['outliers'] = outliers
    stats_summary['outlier_percentage'] = outliers / len(df) * 100
    
    logger.info(f"Calculated statistics for {len(columns)} columns")
    
    return stats_summary.round(4)

def save_results(data: Any, filepath: str, format: str = 'csv', **kwargs):
    """
    Save analysis results in various formats.
    
    Parameters:
    -----------
    data : Any
        Data to save (DataFrame, dict, list, etc.)
    filepath : str
        Path to save file
    format : str
        Output format ('csv', 'excel', 'json', 'pickle')
    **kwargs : dict
        Additional arguments for save functions
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format.lower() == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=kwargs.get('index', False))
            else:
                pd.DataFrame(data).to_csv(filepath, index=kwargs.get('index', False))
                
        elif format.lower() == 'excel':
            if isinstance(data, pd.DataFrame):
                data.to_excel(filepath, index=kwargs.get('index', False))
            elif isinstance(data, dict):
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    for sheet_name, sheet_data in data.items():
                        if isinstance(sheet_data, pd.DataFrame):
                            sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
                        else:
                            pd.DataFrame(sheet_data).to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                pd.DataFrame(data).to_excel(filepath, index=kwargs.get('index', False))
                
        elif format.lower() == 'json':
            if isinstance(data, (pd.DataFrame, pd.Series)):
                data.to_json(filepath, **kwargs)
            else:
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str, **kwargs)
                    
        elif format.lower() == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, **kwargs)
                
        elif format.lower() == 'parquet':
            if isinstance(data, pd.DataFrame):
                data.to_parquet(filepath, **kwargs)
            else:
                pd.DataFrame(data).to_parquet(filepath, **kwargs)
                
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Results saved to {filepath} in {format} format")
        
    except Exception as e:
        logger.error(f"Error saving results to {filepath}: {e}")
        raise

def load_results(filepath: str, format: str = None) -> Any:
    """
    Load saved results.
    
    Parameters:
    -----------
    filepath : str
        Path to saved file
    format : str, optional
        File format (inferred from extension if not provided)
        
    Returns:
    --------
    Any
        Loaded data
    """
    filepath = Path(filepath)
    
    if format is None:
        format = filepath.suffix.lower()[1:]  # Remove dot
        
    try:
        if format == 'csv':
            data = pd.read_csv(filepath)
        elif format == 'excel':
            data = pd.read_excel(filepath)
        elif format == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        elif format == 'parquet':
            data = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Results loaded from {filepath}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading results from {filepath}: {e}")
        raise

def create_report(results: Dict, output_path: str = None, template: str = 'basic'):
    """
    Create a comprehensive analysis report.
    
    Parameters:
    -----------
    results : dict
        Analysis results
    output_path : str, optional
        Path to save report
    template : str
        Report template ('basic', 'detailed', 'executive')
        
    Returns:
    --------
    str
        Report content
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if template == 'executive':
        report = f"""
# DRIVER BEHAVIOR ANALYSIS REPORT
## Executive Summary
**Generated:** {timestamp}

### Key Findings
1. **Driver Segmentation:** {results.get('n_clusters', 'N/A')} distinct behavior clusters identified
2. **Risk Assessment:** {results.get('high_risk_percentage', 0):.1f}% of drivers in high-risk category
3. **Efficiency Score:** Average fuel efficiency: {results.get('avg_efficiency', 0):.2f}
4. **Safety Score:** Average safety rating: {results.get('avg_safety', 0):.2f}

### Recommendations
- Immediate intervention for {results.get('critical_count', 0)} drivers
- Training program for {results.get('training_needed', 0)} drivers
- Estimated fuel savings: {results.get('fuel_savings_potential', 0):.1f}%

### Next Steps
1. Review detailed cluster analysis
2. Implement targeted training programs
3. Monitor progress with monthly reviews
4. Expand analysis to entire fleet
"""
    elif template == 'detailed':
        report = f"""
# DRIVER BEHAVIOR ANALYSIS REPORT
## Detailed Analysis
**Generated:** {timestamp}

### 1. Data Overview
- Total drivers analyzed: {results.get('total_drivers', 0)}
- Data completeness: {results.get('data_completeness', 0):.1f}%
- Analysis period: {results.get('analysis_period', 'N/A')}

### 2. Clustering Results
- Optimal clusters: {results.get('n_clusters', 'N/A')}
- Silhouette score: {results.get('silhouette_score', 0):.3f}
- Davies-Bouldin index: {results.get('davies_bouldin', 0):.3f}

### 3. Risk Distribution
{results.get('risk_distribution', 'N/A')}

### 4. Efficiency Analysis
- Average fuel efficiency: {results.get('avg_efficiency', 0):.3f}
- Best cluster efficiency: {results.get('best_efficiency', 0):.3f}
- Worst cluster efficiency: {results.get('worst_efficiency', 0):.3f}
- Efficiency variability: {results.get('efficiency_variability', 0):.3f}

### 5. Safety Metrics
- Average safety score: {results.get('avg_safety', 0):.3f}
- Harsh acceleration events: {results.get('harsh_accel_events', 0)}
- Harsh braking events: {results.get('harsh_brake_events', 0)}
- Speeding incidents: {results.get('speeding_incidents', 0)}

### 6. Statistical Summary
{results.get('statistical_summary', 'N/A')}

### 7. Recommendations
#### 7.1 Immediate Actions
{results.get('immediate_actions', 'N/A')}

#### 7.2 Short-term Initiatives
{results.get('short_term_initiatives', 'N/A')}

#### 7.3 Long-term Strategies
{results.get('long_term_strategies', 'N/A')}

### 8. Appendices
- Data validation report
- Cluster characteristics
- Driver scorecards
- Technical methodology
"""
    else:  # basic template
        report = f"""
Driver Behavior Analysis Report
===============================
Generated: {timestamp}

Summary:
--------
- Drivers: {results.get('total_drivers', 0)}
- Clusters: {results.get('n_clusters', 'N/A')}
- High Risk: {results.get('high_risk_count', 0)}
- Avg Risk Score: {results.get('avg_risk_score', 0):.1f}
- Fuel Efficiency: {results.get('avg_efficiency', 0):.2f}

Top Findings:
-------------
{chr(10).join([f'- {finding}' for finding in results.get('top_findings', [])])}

Actions Required:
-----------------
{chr(10).join([f'- {action}' for action in results.get('actions_required', [])])}
"""
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")
    
    return report

def visualize_distributions(df: pd.DataFrame, columns: List[str], 
                           output_path: str = None, figsize: Tuple = (15, 10)):
    """
    Create distribution visualizations for multiple columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    columns : list
        Columns to visualize
    output_path : str, optional
        Path to save visualization
    figsize : tuple
        Figure size
    """
    n_cols = min(4, len(columns))
    n_rows = int(np.ceil(len(columns) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for idx, col in enumerate(columns):
        if idx < len(axes):
            ax = axes[idx]
            
            # Plot histogram
            ax.hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            
            # Add vertical lines for statistics
            mean_val = df[col].mean()
            median_val = df[col].median()
            std_val = df[col].std()
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1, alpha=0.7)
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1, alpha=0.7)
            
            ax.set_title(f'Distribution of {col[:30]}...' if len(col) > 30 else f'Distribution of {col}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Distribution visualization saved to {output_path}")
    
    plt.show()
    plt.close()

def export_to_excel(data_dict: Dict[str, pd.DataFrame], output_path: str):
    """
    Export multiple DataFrames to Excel with formatting.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of {sheet_name: DataFrame}
    output_path : str
        Path to save Excel file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                # Truncate sheet name if too long
                sheet_name = str(sheet_name)[:31]
                
                # Write DataFrame to Excel
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Auto-adjust column widths
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        logger.info(f"Data exported to Excel: {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        raise

def generate_dashboard_data(df: pd.DataFrame, config: Dict = None) -> Dict:
    """
    Generate data optimized for dashboard visualization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data with analysis results
    config : dict, optional
        Configuration for dashboard generation
        
    Returns:
    --------
    dict
        Dashboard-ready data
    """
    if config is None:
        config = {
            'top_n_clusters': 5,
            'risk_thresholds': [30, 60, 80],
            'efficiency_bins': 5
        }
    
    dashboard_data = {}
    
    # 1. Summary statistics
    dashboard_data['summary'] = {
        'total_drivers': len(df),
        'avg_risk_score': df.get('overall_risk_score', pd.Series([0])).mean(),
        'avg_efficiency': df.get('fuel_efficiency', pd.Series([0])).mean(),
        'high_risk_count': df[df.get('overall_risk_score', 0) > 70].shape[0] if 'overall_risk_score' in df.columns else 0,
        'harsh_events_total': df.get('harsh_acceleration_count', pd.Series([0])).sum() + 
                             df.get('harsh_braking_count', pd.Series([0])).sum()
    }
    
    # 2. Cluster distribution
    if 'cluster' in df.columns:
        cluster_dist = df['cluster'].value_counts().sort_index()
        dashboard_data['clusters'] = {
            'labels': [f'Cluster {i}' for i in cluster_dist.index],
            'counts': cluster_dist.values.tolist(),
            'percentages': (cluster_dist.values / len(df) * 100).round(1).tolist()
        }
    
    # 3. Risk distribution
    if 'overall_risk_score' in df.columns:
        risk_bins = pd.cut(df['overall_risk_score'], 
                          bins=[0, 30, 60, 80, 100],
                          labels=['Low', 'Moderate', 'High', 'Critical'])
        risk_dist = risk_bins.value_counts()
        dashboard_data['risk_distribution'] = {
            'categories': risk_dist.index.tolist(),
            'counts': risk_dist.values.tolist()
        }
    
    # 4. Efficiency distribution
    if 'fuel_efficiency' in df.columns:
        eff_bins = pd.cut(df['fuel_efficiency'], bins=config['efficiency_bins'])
        eff_dist = eff_bins.value_counts().sort_index()
        dashboard_data['efficiency_distribution'] = {
            'bins': [str(bin) for bin in eff_dist.index],
            'counts': eff_dist.values.tolist()
        }
    
    # 5. Top drivers by risk
    if 'overall_risk_score' in df.columns and 'driver_id' in df.columns:
        top_risky = df.nlargest(10, 'overall_risk_score')[['driver_id', 'overall_risk_score']]
        dashboard_data['top_risky_drivers'] = {
            'driver_ids': top_risky['driver_id'].tolist(),
            'scores': top_risky['overall_risk_score'].round(1).tolist()
        }
    
    # 6. Correlation matrix (top features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        # Select top features for correlation
        selected_features = numeric_cols[:min(10, len(numeric_cols))]
        corr_matrix = df[selected_features].corr().round(3)
        
        # Convert to list format for visualization
        dashboard_data['correlation_matrix'] = {
            'features': selected_features,
            'matrix': corr_matrix.values.tolist()
        }
    
    # 7. Time-series data (if available)
    time_cols = [col for col in df.columns if 'time' in col.lower()]
    if time_cols:
        dashboard_data['time_metrics'] = {
            'metrics': time_cols[:5],
            'average_values': df[time_cols[:5]].mean().round(3).tolist()
        }
    
    logger.info("Dashboard data generated successfully")
    
    return dashboard_data

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format float as percentage string."""
    return f"{value:.{decimals}f}%"

def format_currency(value: float, currency: str = "$") -> str:
    """Format float as currency string."""
    if abs(value) >= 1_000_000:
        return f"{currency}{value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"{currency}{value/1_000:.1f}K"
    else:
        return f"{currency}{value:.2f}"

def safe_divide(numerator, denominator, default=0):
    """Safe division with default value for zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator

def detect_anomalies(df: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect anomalies in numerical data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    method : str
        Detection method ('iqr', 'zscore', 'modified_zscore')
    threshold : float
        Detection threshold
        
    Returns:
    --------
    pd.DataFrame
        Boolean mask of anomalies
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    anomalies = pd.DataFrame(False, index=df.index, columns=numeric_cols)
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        if method == 'iqr':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            anomalies[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        elif method == 'zscore':
            mean = data.mean()
            std = data.std()
            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                anomalies[col] = z_scores > threshold
                
        elif method == 'modified_zscore':
            median = data.median()
            mad = stats.median_abs_deviation(data, scale='normal')
            if mad > 0:
                modified_z = 0.6745 * (df[col] - median) / mad
                anomalies[col] = np.abs(modified_z) > threshold
    
    logger.info(f"Anomalies detected using {method} method")
    return anomalies

def calculate_performance_metrics(y_true, y_pred, labels=None):
    """
    Calculate comprehensive performance metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list, optional
        Label names
        
    Returns:
    --------
    dict
        Performance metrics
    """
    metrics = {}
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, labels=labels)
    metrics['classification_report'] = report
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Additional metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
    
    logger.info(f"Performance metrics calculated: Accuracy = {metrics['accuracy']:.3f}")
    
    return metrics

def get_memory_usage(df: pd.DataFrame) -> Dict:
    """
    Get memory usage information for DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    dict
        Memory usage information
    """
    memory_info = {}
    
    # Total memory usage
    memory_info['total_memory_mb'] = df.memory_usage(deep=True).sum() / 1024**2
    
    # Memory by data type
    memory_by_dtype = {}
    for dtype in df.dtypes.unique():
        cols = df.select_dtypes(include=[dtype]).columns
        memory_by_dtype[str(dtype)] = df[cols].memory_usage(deep=True).sum() / 1024**2
    
    memory_info['memory_by_dtype'] = memory_by_dtype
    
    # Top memory-consuming columns
    memory_per_col = df.memory_usage(deep=True)
    top_cols = memory_per_col.nlargest(10)
    memory_info['top_memory_cols'] = {
        'columns': top_cols.index.tolist(),
        'memory_mb': (top_cols.values / 1024**2).tolist()
    }
    
    # Memory optimization suggestions
    suggestions = []
    
    # Check for object dtype columns that could be categorical
    object_cols = df.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        for col in object_cols:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # If less than 50% unique values
                suggestions.append(f"Column '{col}' could be converted to categorical (unique ratio: {unique_ratio:.1%})")
    
    # Check for float columns that could be downcast
    float_cols = df.select_dtypes(include=['float']).columns
    for col in float_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min >= -32768 and col_max <= 32767 and df[col].dtype == 'float64':
            suggestions.append(f"Column '{col}' could be downcast to float32")
    
    memory_info['optimization_suggestions'] = suggestions
    
    logger.info(f"Memory usage: {memory_info['total_memory_mb']:.2f} MB")
    
    return memory_info