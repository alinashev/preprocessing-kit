import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder

from config import ORDINAL_MAPPINGS, ONE_HOT_COLS, CATEGORY_MAPPINGS
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
                 positive_target_label="yes"):
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

    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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
        df = remove_unnecessary_columns(df, self.unnecessary_columns)
        X_train, X_val, y_train, y_val = split_data(df, self.target_col, self.test_size, self.random_state)

        numeric_cols = define_numerical_cols(df)

        # Ordinal Encoding
        for col, categories in ORDINAL_MAPPINGS.items():
            X_train, encoder = ordinal_encode(X_train, col, categories)
            X_val[col] = encoder.transform(X_val[[col]])
            self.ordinal_encoders[col] = encoder

        # One-Hot Encoding
        X_train, one_hot_encoder, encoded_cols = one_hot_encode(X_train, ONE_HOT_COLS)
        X_val[encoded_cols] = one_hot_encoder.transform(X_val[ONE_HOT_COLS])
        X_val = X_val.drop(columns=ONE_HOT_COLS)
        self.one_hot_encoder = one_hot_encoder
        self.encoded_cols = encoded_cols

        # Binary Encoding
        X_train['contact'] = X_train['contact'].map(define_binary_codes(X_train['contact']))
        X_val['contact'] = X_val['contact'].map(define_binary_codes(X_val['contact']))

        # Map Categorical Values
        for col, mapping in CATEGORY_MAPPINGS.items():
            X_train[col] = X_train[col].map(mapping)
            X_val[col] = X_val[col].map(mapping)

        # Feature Scaling
        if self.scaler_numeric:
            if self.scaling_method == "minmax":
                X_train[numeric_cols], scaler = min_max_scaler(X_train, numeric_cols)
                X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
            else:
                X_train[numeric_cols], scaler = standard_scaler(X_train, numeric_cols)
                X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
            self.scaler = scaler

        self.input_cols = list(X_train.columns)

        y_train, y_val = self.__encode_target(y_train, y_val)

        return X_train, y_train, X_val, y_val
