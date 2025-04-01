import numpy as np
import pandas as pd


class OutlierProcessor:
    """
    A class for handling outliers in a dataset using configurable methods.

    Attributes:
        df (pd.DataFrame): The dataset to process.
        config (dict): A dictionary specifying the outlier processing strategies for each column.
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        """
        Initializes the OutlierProcessor with a dataset and configuration settings.

        Args:
            df (pd.DataFrame): The input dataset.
            config (dict): Configuration dictionary specifying outlier processing methods.
        """
        self.df = df
        self.config = config

    def clip_values(self, col: str, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        """
        Clips outliers in the specified column based on quantile thresholds.

        Args:
            col (str): Column name to process.
            lower_quantile (float, optional): Lower quantile threshold. Defaults to 0.01.
            upper_quantile (float, optional): Upper quantile threshold. Defaults to 0.99.
        """
        lower_bound = self.df[col].quantile(lower_quantile)
        upper_bound = self.df[col].quantile(upper_quantile)
        self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)

    def drop_column(self, col: str):
        """
        Drops the specified column from the dataset if it exists.

        Args:
            col (str): Column name to drop.
        """
        if col in self.df.columns:
            self.df.drop(columns=[col], inplace=True)

    def apply_log_transform(self, col: str):
        """
        Applies a log transformation to the specified column to normalize skewed distributions.

        Args:
            col (str): Column name to transform.
        """
        self.df[col] = np.log1p(self.df[col])

    def create_binary_feature(self, col: str, threshold: float, new_col: str):
        """
        Creates a binary feature based on a threshold value.

        Args:
            col (str): Column name to process.
            threshold (float): Value used to create the binary feature.
            new_col (str): Name of the new binary column.
        """
        self.df[new_col] = (self.df[col] != threshold).astype(int)
        self.df.drop(columns=[col], inplace=True)

    def preprocess(self) -> pd.DataFrame:
        """
        Performs outlier processing based on the provided configuration.

        Steps applied:
            - Clips values in specified columns based on quantiles.
            - Drops specified columns from the dataset.
            - Applies log transformation to specified columns.
            - Creates binary features based on specified thresholds.

        Returns:
            pd.DataFrame: The dataset with outliers processed.
        """
        if "clip_columns" in self.config:
            for col, params in self.config["clip_columns"].items():
                self.clip_values(col, **params)

        if "drop_columns" in self.config:
            for col in self.config["drop_columns"]:
                self.drop_column(col)

        if "log_transform" in self.config:
            for col in self.config["log_transform"]:
                self.apply_log_transform(col)

        if "binary_features" in self.config:
            for col, params in self.config["binary_features"].items():
                self.create_binary_feature(col, params["threshold"], params["new_col"])

        return self.df
