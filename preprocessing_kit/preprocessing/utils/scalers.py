"""
scalers.py

This module provides functions for scaling numerical features in a DataFrame.
It includes Min-Max scaling and Standard scaling.

Functions:
- min_max_scaler: Apply Min-Max scaling to specified numeric columns.
- standard_scaler: Apply Standard scaling (z-score normalization) to specified numeric columns.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List, Tuple


def min_max_scaler(
        df: pd.DataFrame, numeric_cols: List[str]
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Apply Min-Max scaling to specified numeric columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numeric_cols (List[str]): List of numeric columns to scale.

    Returns:
        Tuple[pd.DataFrame, MinMaxScaler]:
            Transformed DataFrame with scaled columns and the fitted scaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(df[numeric_cols])
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df[numeric_cols], scaler


def standard_scaler(
        df: pd.DataFrame, numeric_cols: List[str]
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Apply Standard scaling (z-score normalization) to specified numeric columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numeric_cols (List[str]): List of numeric columns to scale.

    Returns:
        Tuple[pd.DataFrame, StandardScaler]:
            Transformed DataFrame with scaled columns and the fitted scaler.
    """
    scaler = StandardScaler()
    scaler.fit(df[numeric_cols])
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df[numeric_cols], scaler
