import pandas as pd


class FeatureEngineer:
    """
    Class for creating new features based on a given configuration.

    Attributes:
        df (pd.DataFrame): The dataset to transform.
        config (dict): A dictionary specifying which features to generate.
    """
    def __init__(self, df: pd.DataFrame, config: dict):
        """
        Initializes the FeatureEngineer with the dataset and configuration.

        Args:
            df (pd.DataFrame): The dataset to transform.
            config (dict): Configuration dictionary specifying feature transformations.
        """
        self.df = df
        self.config = config

    def interaction_features(self) -> pd.DataFrame:
        """
        Creates interaction features by multiplying specified columns.

        Returns:
            pd.DataFrame: The updated dataset with interaction features.
        """
        for col1, col2 in self.config.get("INTERACTION_FEATURES", []):
            self.df[f"{col1}_{col2}_interaction"] = self.df[col1] * self.df[col2]
        return self.df

    def ratio_features(self) -> pd.DataFrame:
        """
        Creates ratio features by dividing specified columns.

        Returns:
            pd.DataFrame: The updated dataset with ratio features.
        """
        for col1, col2 in self.config.get("RATIO_FEATURES", []):
            if col1 in self.df.columns and col2 in self.df.columns:
                self.df[f"{col1}_{col2}_ratio"] = self.df[col1] / (self.df[col2] + 1)
        return self.df

    def economic_features(self) -> pd.DataFrame:
        """
        Creates economic-related features by multiplying specified columns.

        Returns:
            pd.DataFrame: The updated dataset with economic features.
        """
        for col1, col2 in self.config.get("ECONOMIC_FEATURES", []):
            self.df[f"{col1}_{col2}_interaction"] = self.df[col1] * self.df[col2]
        return self.df

    def time_features(self) -> pd.DataFrame:
        """
        Creates time-based features using specified conditions.

        Returns:
            pd.DataFrame: The updated dataset with time features.
        """
        for new_col, details in self.config.get("TIME_FEATURES", {}).items():
            col, values = details["column"], details["values"]
            self.df[new_col] = self.df[col].apply(lambda x: 1 if x in values else 0)
        return self.df

    def contact_features(self) -> pd.DataFrame:
        """
        Creates features related to contact information.

        Returns:
            pd.DataFrame: The updated dataset with contact features.
        """
        if "total_contacts" in self.config.get("CONTACT_FEATURES", []):
            self.df["total_contacts"] = self.df["previous"] + self.df["campaign"]
        if "was_previously_contacted" in self.config.get("CONTACT_FEATURES", []):
            self.df["was_previously_contacted"] = (self.df["previous"] > 0).astype(int)
        if "campaign_to_total_ratio" in self.config.get("CONTACT_FEATURES", []):
            self.df["campaign_to_total_ratio"] = self.df["campaign"] / (self.df["total_contacts"] + 1)
        return self.df

    def preprocess(self) -> pd.DataFrame:
        """
        Executes feature generation based on the provided configuration.

        Returns:
            pd.DataFrame: The transformed dataset with newly created features.
        """
        self.interaction_features()
        self.ratio_features()
        self.economic_features()
        self.time_features()
        self.contact_features()
        return self.df
