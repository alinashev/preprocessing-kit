"""
utils.py

This module provides functions for preprocessing data in preparation for machine learning tasks.
It includes functions for splitting data, identifying numerical and categorical columns, and removing unnecessary columns.

Functions:
- split_data: Split the dataset into training and validation sets.
- define_numerical_cols: Identify numerical columns in a DataFrame.
- define_categorical_cols: Identify categorical columns in a DataFrame.
- remove_unnecessary_columns: Remove specified unnecessary columns from the DataFrame.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame, target_col: str, test_size: float, random_state: int) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and validation sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.
        test_size (float): The proportion of the data to be used for the validation set.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            Training data (X_train), validation data (X_val), training labels (y_train), and validation labels (y_val).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def define_numerical_cols(df: pd.DataFrame) -> List[str]:
    """
    Identify numerical columns in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        List[str]: A list of column names that are numerical.
    """
    return df.select_dtypes(include=np.number).columns.tolist()


def define_categorical_cols(df: pd.DataFrame) -> List[str]:
    """
    Identify categorical columns in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        List[str]: A list of column names that are categorical.
    """
    return df.select_dtypes('object').columns.tolist()


def remove_unnecessary_columns(df: pd.DataFrame, columns_to_remove: List[str]) -> pd.DataFrame:
    """
    Remove unnecessary columns from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_remove (List[str]): A list of column names to be removed from the DataFrame.

    Returns:
        pd.DataFrame: The DataFrame without the specified unnecessary columns.
    """
    return df.drop(columns=columns_to_remove, errors="ignore")
