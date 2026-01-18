"""
Driver Behavior Analysis Package

A comprehensive system for analyzing driving patterns, 
identifying risky behaviors, and optimizing fleet efficiency.

Modules:
- data_processor: Data loading, cleaning, and basic processing
- feature_extractor: Advanced feature engineering
- clustering: Driver segmentation using ML algorithms
- utils: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "Driver Behavior Analysis Team"
__email__ = "your.email@example.com"

from .data_processor import DataProcessor
from .feature_extractor import FeatureExtractor
from .clustering import DriverClustering
from .utils import (
    save_results,
    load_config,
    create_report,
    validate_data,
    calculate_statistics,
    visualize_distributions,
    export_to_excel,
    generate_dashboard_data
)

# Define what gets imported with "from src import *"
__all__ = [
    'DataProcessor',
    'FeatureExtractor', 
    'DriverClustering',
    'save_results',
    'load_config',
    'create_report',
    'validate_data',
    'calculate_statistics',
    'visualize_distributions',
    'export_to_excel',
    'generate_dashboard_data'
]

# Package metadata
PACKAGE_INFO = {
    "name": "driver_behavior_analysis",
    "version": __version__,
    "description": "A comprehensive system for analyzing driving patterns and optimizing fleet efficiency",
    "author": __author__,
    "dependencies": [
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "pyyaml>=5.4.0"
    ],
    "license": "Proprietary",
    "status": "Production"
}

# Configuration defaults
DEFAULT_CONFIG = {
    "data": {
        "input_file": "data/data_cleaned.csv",
        "output_dir": "results/",
        "cache_dir": ".cache/"
    },
    "clustering": {
        "n_clusters": 5,
        "random_state": 42,
        "max_iter": 300,
        "n_init": 10
    },
    "thresholds": {
        "harsh_acceleration": 3.5,
        "harsh_braking": -3.5,
        "speeding": 88,
        "dangerous_rpm": 3000
    },
    "visualization": {
        "color_palette": "viridis",
        "style": "seaborn",
        "save_format": "png",
        "dpi": 300
    }
}

def get_version():
    """Return the package version."""
    return __version__

def get_info():
    """Return package information dictionary."""
    return PACKAGE_INFO.copy()

def setup_logging(level="INFO"):
    """
    Setup logging configuration for the package.
    
    Parameters:
    -----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    import logging
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('driver_analysis.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Driver Behavior Analysis Package v{__version__} initialized")
    
    return logger

def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
    --------
    dict
        Dictionary with dependency check results
    """
    import importlib.util
    import sys
    
    dependencies = PACKAGE_INFO["dependencies"]
    results = {}
    
    for dep in dependencies:
        try:
            # Parse package name and version
            if ">=" in dep:
                pkg_name, min_version = dep.split(">=")
            else:
                pkg_name, min_version = dep, None
            
            # Try to import
            spec = importlib.util.find_spec(pkg_name.split("[")[0])
            if spec is None:
                results[dep] = {"installed": False, "error": "Package not found"}
            else:
                if min_version:
                    # Try to get version
                    try:
                        module = importlib.import_module(pkg_name)
                        installed_version = getattr(module, "__version__", "unknown")
                        results[dep] = {
                            "installed": True,
                            "version": installed_version,
                            "min_version": min_version,
                            "ok": installed_version >= min_version if installed_version != "unknown" else True
                        }
                    except:
                        results[dep] = {"installed": True, "version": "unknown"}
                else:
                    results[dep] = {"installed": True}
        except Exception as e:
            results[dep] = {"installed": False, "error": str(e)}
    
    return results

def initialize(config_path=None):
    """
    Initialize the driver behavior analysis system.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to configuration file
        
    Returns:
    --------
    tuple
        (config, logger, dependencies_check)
    """
    import logging
    
    # Setup logging
    logger = setup_logging()
    
    # Check dependencies
    deps = check_dependencies()
    missing = [k for k, v in deps.items() if not v.get("installed", False)]
    if missing:
        logger.warning(f"Missing dependencies: {missing}")
    
    # Load configuration
    if config_path:
        from .utils import load_config
        config = load_config(config_path)
    else:
        config = DEFAULT_CONFIG.copy()
    
    logger.info("Driver Behavior Analysis system initialized successfully")
    
    return config, logger, deps

# Create a default logger when module is imported
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())