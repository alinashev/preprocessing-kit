"""
Module for processing and transforming datasets.

This module provides the `DataProcessor` class, which applies various preprocessing steps,
outlier handling, and feature engineering to a dataset before feeding it into a machine learning model.

Dependencies:
    - pandas
    - enrich.features.FeatureEngineer
    - preprocessing.missing_value_handler.MissingValueHandler
    - preprocessing.outlier_processor.OutlierProcessor
    - preprocessing.processor.Processor
    - utils.config (OUTLIER_CONFIG, FEATURE_ENGINEERING_CONFIG, MISSING_VALUE_CONFIG)
"""

import pandas as pd

from config import MISSING_VALUE_CONFIG, OUTLIER_CONFIG, FEATURE_ENGINEERING_CONFIG
from preprocessing_kit.enrichment.features import FeatureEngineer
from preprocessing_kit.preprocessing.missing_value_handler import MissingValueHandler
from preprocessing_kit.preprocessing.outlier_processor import OutlierProcessor
from preprocessing_kit.preprocessing.processor import Processor


class DataProcessor:
    """
    A class for preprocessing datasets, handling missing values, outliers, and feature engineering.

    Attributes:
        processor (Processor): An instance of the Processor class for applying transformations.
        handle_unknowns (bool): Whether to handle unknown values (default: True).
        apply_preprocessing (bool): Whether to apply general preprocessing (default: True).
        apply_outliers (bool): Whether to process outliers (default: False).
        apply_feature_engineering (bool): Whether to apply feature engineering (default: False).

        missing_value_config (dict)
        outlier_config(dict)
        outlier_config(dict)
        feature_engineering_config(dict)

    """

    def __init__(self, processor: Processor,
                 handle_unknowns: bool = True,
                 apply_preprocessing: bool = True,
                 apply_outliers_processing: bool = False,
                 apply_feature_engineering: bool = False
                 ):
        """
        Initializes the DataProcessor instance with specified preprocessing configurations.

        Args:
            processor (Processor): Instance of a processor to apply transformations.
            handle_unknowns (bool, optional): Whether to handle unknown values. Defaults to True.
            apply_preprocessing (bool, optional): Whether to apply preprocessing. Defaults to True.
            apply_outliers_processing (bool, optional): Whether to process outliers. Defaults to False.
            apply_feature_engineering (bool, optional): Whether to apply feature engineering. Defaults to False.
        """
        self.processor = processor
        self.handle_unknowns = handle_unknowns
        self.apply_outliers = apply_outliers_processing
        self.apply_feature_engineering = apply_feature_engineering
        self.apply_preprocessing = apply_preprocessing

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values in the dataset, specifically replacing 'unknown' values in the 'education' column.

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with missing values handled.
        """
        if "education" in df.columns:
            mode_value = df["education"][df["education"] != "unknown"].mode()[0]
            df["education_unknown"] = (df["education"] == "unknown").astype(int)
            df["education"] = df["education"].replace("unknown", mode_value)
        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies preprocessing steps based on the provided configurations.

        Steps applied:
            - Handles missing values (if enabled).
            - Processes outliers (if enabled).
            - Applies feature engineering (if enabled).
            - Runs final transformations using the assigned processor.

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The processed dataset.
        """
        if self.handle_unknowns:
            missing_handler = MissingValueHandler(df, MISSING_VALUE_CONFIG)
            df = missing_handler.preprocess()

        if self.handle_unknowns:
            df = self.handle_missing_values(df)

        if self.apply_outliers:
            outlier_processor = OutlierProcessor(df, OUTLIER_CONFIG)
            df = outlier_processor.preprocess()

        if self.apply_feature_engineering:
            feature_engineer = FeatureEngineer(df, FEATURE_ENGINEERING_CONFIG)
            df = feature_engineer.preprocess()

        if self.apply_preprocessing:
            return self.processor.process(df)

        return df
