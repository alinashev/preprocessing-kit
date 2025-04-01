import pandas as pd


class MissingValueHandler:
    """
    A class for handling missing values in a dataset using various strategies.

    Attributes:
        df (pd.DataFrame): The dataset to process.
        config (dict): A dictionary specifying the strategy for handling missing values per column.
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        """
        Initializes the MissingValueHandler with the dataset and configuration.

        Args:
            df (pd.DataFrame): The input dataset.
            config (dict): Dictionary specifying the strategy for handling missing values per column.
        """
        self.df = df
        self.config = config

    def fill_missing_values(self, col: str, strategy: str, fill_value=None):
        """
        Fills missing values in a specified column based on the given strategy.

        Args:
            col (str): The column name.
            strategy (str): The strategy to use for filling missing values. Options include:
                - "mode": Replace "unknown" values with the most frequent value.
                - "median": Replace missing values with the median.
                - "mean": Replace missing values with the mean.
                - "constant": Replace "unknown" values with a specified constant value.
                - "skip": Do not modify the column.
            fill_value (optional): The value to use if the strategy is "constant".

        Raises:
            ValueError: If an unsupported strategy is provided.
        """
        if strategy == "mode":
            mode_value = self.df[col][self.df[col] != "unknown"].mode()[0]
            self.df[col] = self.df[col].replace("unknown", mode_value)
        elif strategy == "median":
            median_value = self.df[col].median()
            self.df[col] = self.df[col].fillna(median_value)
        elif strategy == "mean":
            mean_value = self.df[col].mean()
            self.df[col] = self.df[col].fillna(mean_value)
        elif strategy == "constant":
            self.df[col] = self.df[col].replace("unknown", fill_value)
        elif strategy == "skip":
            pass  # Do nothing
        else:
            raise ValueError(f"Unsupported filling strategy: {strategy}")

    def preprocess(self) -> pd.DataFrame:
        """
        Processes all missing values based on the provided configuration.

        Steps performed:
            - Creates binary indicators for "unknown" categorical values.
            - Applies the specified filling strategy to handle missing values.

        Returns:
            pd.DataFrame: The dataset with missing values handled.
        """
        for col, params in self.config.items():
            if col in self.df.columns:
                if self.df[col].dtype == "object":
                    self.df[f"{col}_unknown"] = (self.df[col] == "unknown").astype(int)

                strategy = params["strategy"]
                fill_value = params.get("fill_value", None)
                self.fill_missing_values(col, strategy, fill_value)

        return self.df
