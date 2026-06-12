import logging
import time
import functools
from pathlib import Path
from typing import Optional, Any
import pandas as pd
import numpy as np
import joblib

# Set up logging configuration with optional file logging
def setup_logger(name: str = 'airline_loyalty', log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

# Globally initialized logger
logger = setup_logger()

# Decorator to measure function execution time
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info(f'Starting: {func.__name__}')
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f'Completed: {func.__name__} in {elapsed:.2f}s')
        return result
    return wrapper

# Read CSV dataset with logging
def load_csv(filepath: str | Path, **kwargs) -> pd.DataFrame:
    filepath = Path(filepath)
    logger.info(f'Loading: {filepath.name}')
    df = pd.read_csv(filepath, **kwargs)
    logger.info(f'Loaded {filepath.name}: {df.shape[0]:,} rows × {df.shape[1]} cols')
    return df

# Save DataFrame to CSV with logging
def save_csv(df: pd.DataFrame, filepath: str | Path, index: bool = False, **kwargs):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index, **kwargs)
    logger.info(f'Saved: {filepath.name} ({df.shape[0]:,} rows × {df.shape[1]} cols)')

# Serialize Python object/model to disk
def save_model(model: Any, filepath: str | Path):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f'Saved model: {filepath.name}')

# Load serialized model from disk
def load_model(filepath: str | Path) -> Any:
    filepath = Path(filepath)
    logger.info(f'Loading model: {filepath.name}')
    return joblib.load(filepath)

# Generate profile metrics summary of a DataFrame
def describe_df(df: pd.DataFrame, name: str = 'DataFrame') -> dict:
    return {
        'name': name,
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / 1000000.0,
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicated_rows': df.duplicated().sum()
    }

# Print a visual text summary of a DataFrame's properties
def print_summary(summary: dict):
    print(f"\n{'=' * 50}")
    print(f"  {summary['name']}")
    print(f"{'=' * 50}")
    print(f"  Rows: {summary['shape'][0]:,}")
    print(f"  Columns: {summary['shape'][1]}")
    print(f"  Memory: {summary['memory_mb']:.2f} MB")
    print(f"  Duplicated Rows: {summary['duplicated_rows']:,}")
    print(f"  Column Types: {summary['dtypes']}")
    print(f'\n  Missing Values (%):')
    for col, pct in summary['missing_pct'].items():
        if pct > 0:
            print(f'    {col}: {pct:.1f}%')
    print(f"{'=' * 50}\n")

# Cap extreme values to specified percentiles
def cap_outliers(series: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.Series:
    lower = series.quantile(lower_pct)
    upper = series.quantile(upper_pct)
    return series.clip(lower, upper)

# Compute IQR bounds to detect outliers
def iqr_outlier_bounds(series: pd.Series, multiplier: float = 1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (lower, upper)

# Min-max scale a Series between 0 and 1
def normalize_minmax(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)