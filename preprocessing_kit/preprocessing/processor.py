import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder

from configuration.configuration import Configuration
from .utils.encoders import ordinal_encode, one_hot_encode, define_binary_codes
from .utils.scalers import min_max_scaler, standard_scaler

from .utils.utils import remove_unnecessary_columns, split_data, define_numerical_cols


class Processor:
    """
    A class for handling data preprocessing tasks such as encoding categorical variables,
    scaling numerical features, and splitting data into training and validation sets.

    Attributes:
        target_col (str): The name of the target column.
        unnecessary_columns (List[str]): A list of columns to be removed before processing.
        test_size (float): The proportion of data to be used for validation.
        random_state (int): The random seed for reproducibility.
        scaler_numeric (bool): Whether to scale numerical features.
        scaling_method (str): The method to use for scaling ("standard" or "minmax").
        positive_target_label (str): The label to be treated as the positive class in binary classification.
        ordinal_encoders (Dict[str, OrdinalEncoder]): Dictionary storing ordinal encoders for categorical features.
        one_hot_encoder (Optional[OneHotEncoder]): OneHotEncoder instance for categorical encoding.
        scaler (Optional[MinMaxScaler | StandardScaler]): Scaler instance for numerical feature scaling.
        input_cols (List[str]): List of input feature names after preprocessing.
        encoded_cols (List[str]): List of new columns created through one-hot encoding.
    """

    def __init__(self, target_col: str, unnecessary_columns: Optional[List[str]] = None, test_size: float = 0.2,
                 random_state: int = 42, scaler_numeric: bool = True, scaling_method: str = "standard",
                 positive_target_label="yes", config: Optional[Configuration] = None):
        """
        Initializes the Processor class with configuration settings.

        Parameters:
            target_col (str): The target column name.
            unnecessary_columns (Optional[List[str]]): List of columns to be removed.
            test_size (float): The fraction of data used for validation.
            random_state (int): Seed for reproducibility.
            scaler_numeric (bool): Whether to apply scaling to numeric columns.
            scaling_method (str): The scaling method ("standard" or "minmax").
            positive_target_label (str): Label considered as "positive" in binary classification.
        """
        self.target_col = target_col
        self.unnecessary_columns = unnecessary_columns if unnecessary_columns else []
        self.test_size = test_size
        self.random_state = random_state
        self.scaler_numeric = scaler_numeric
        self.scaling_method = scaling_method
        self.positive_target_label = positive_target_label
        self.config: Configuration = config

        self.ordinal_encoders: Dict[str, OrdinalEncoder] = {}
        self.one_hot_encoder: Optional[OneHotEncoder] = None
        self.scaler: Optional[MinMaxScaler | StandardScaler] = None
        self.input_cols: List[str] = []
        self.encoded_cols: List[str] = []

    def __encode_target(self, y_train: pd.Series, y_val: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Encodes the target variable as binary (1 for the positive class, 0 otherwise).

        Parameters:
            y_train (pd.Series): The training target variable.
            y_val (pd.Series): The validation target variable.

        Returns:
            Tuple[pd.Series, pd.Series]: Encoded y_train and y_val.
        """
        y_train = (y_train == self.positive_target_label).astype(int)
        y_val = (y_val == self.positive_target_label).astype(int)
        return y_train, y_val

    def get_encoded_feature_indices(self, numeric_cols: List[str]) -> List[int]:
        """
        Retrieves the indices of the encoded categorical features in the processed dataset.

        Parameters:
            numeric_cols (List[str]): List of numerical column names.

        Returns:
            List[int]: List of indices of categorical features after encoding.
        """
        return [self.input_cols.index(col) for col in self.input_cols if col not in numeric_cols]

    def _apply_ordinal_encoding(self, x_train: pd.DataFrame, x_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for col, categories in self.config.encoding_config.ordinal_encoder.items():
            x_train, encoder = ordinal_encode(x_train, col, categories)
            x_val[col] = encoder.transform(x_val[[col]])
            self.ordinal_encoders[col] = encoder
        return x_train, x_val

    def _apply_one_hot_encoding(self, x_train: pd.DataFrame, x_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        x_train, one_hot_encoder, encoded_cols = one_hot_encode(x_train, self.config.encoding_config.one_hot_encoder)
        x_val[encoded_cols] = one_hot_encoder.transform(x_val[self.config.encoding_config.one_hot_encoder])
        x_val = x_val.drop(columns=self.config.encoding_config.one_hot_encoder)
        self.one_hot_encoder = one_hot_encoder
        self.encoded_cols = encoded_cols
        return x_train, x_val

    def _apply_binary_encoding(self, x_train: pd.DataFrame, x_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for col in self.config.encoding_config.binary_encoder:
            if col in x_train.columns:
                x_train[col] = x_train[col].map(define_binary_codes(x_train[col]))
            if col in x_val.columns:
                x_val[col] = x_val[col].map(define_binary_codes(x_val[col]))
        return x_train, x_val

    def _apply_category_mapping(self, x_train: pd.DataFrame, x_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for col, mapping in self.config.encoding_config.category_mappings.items():
            if col in x_train.columns:
                x_train[col] = x_train[col].map(mapping)
            if col in x_val.columns:
                x_val[col] = x_val[col].map(mapping)
        return x_train, x_val

    def _apply_scaling(self, x_train: pd.DataFrame, x_val: pd.DataFrame, numeric_cols: List[str]) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        if self.scaling_method == "minmax":
            x_train[numeric_cols], scaler = min_max_scaler(x_train, numeric_cols)
            x_val[numeric_cols] = scaler.transform(x_val[numeric_cols])
        else:
            x_train[numeric_cols], scaler = standard_scaler(x_train, numeric_cols)
            x_val[numeric_cols] = scaler.transform(x_val[numeric_cols])
        self.scaler = scaler
        return x_train, x_val

    def _remove_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return remove_unnecessary_columns(df, self.unnecessary_columns)

    def _split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return split_data(df, self.target_col, self.test_size, self.random_state)

    def _separate_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        return X, X.copy(), y, y.copy()

    def _encode_features(self, x_train: pd.DataFrame, x_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.config.encoding_config.ordinal_encoder:
            x_train, x_val = self._apply_ordinal_encoding(x_train, x_val)

        if self.config.encoding_config.one_hot_encoder:
            x_train, x_val = self._apply_one_hot_encoding(x_train, x_val)

        if self.config.encoding_config.binary_encoder:
            x_train, x_val = self._apply_binary_encoding(x_train, x_val)

        if self.config.encoding_config.category_mappings:
            x_train, x_val = self._apply_category_mapping(x_train, x_val)

        return x_train, x_val

    def _scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, numeric_cols: List[str]) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        if self.scaler_numeric:
            return self._apply_scaling(X_train, X_val, numeric_cols)
        return X_train, X_val

    def process(self, df: pd.DataFrame, split: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Processes the given DataFrame by performing the following steps:
        1. Removes unnecessary columns.
        2. Splits the data into training and validation sets.
        3. Applies ordinal encoding to categorical variables with predefined order.
        4. Applies one-hot encoding to categorical variables.
        5. Maps categorical string values to numerical values.
        6. Applies feature scaling if enabled.

        Parameters:
            df (pd.DataFrame): The input dataset.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
                - X_train (pd.DataFrame): Processed training features.
                - y_train (pd.Series): Encoded training target.
                - X_val (pd.DataFrame): Processed validation features.
                - y_val (pd.Series): Encoded validation target.
        """
        df = self._remove_columns(df)

        X_train, X_val, y_train, y_val = self._split(df) if split else self._separate_features_target(df)

        numeric_cols = define_numerical_cols(df)

        X_train, X_val = self._encode_features(X_train, X_val)
        X_train, X_val = self._scale_features(X_train, X_val, numeric_cols)

        self.input_cols = list(X_train.columns)

        y_train, y_val = self.__encode_target(y_train, y_val)

        return X_train, y_train, X_val, y_val
