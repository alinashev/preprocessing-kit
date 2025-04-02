import pandas as pd

from preprocessing_kit.preprocessing.encoders.base_encoder import BaseEncoder
from preprocessing_kit.preprocessing.utils.encoders import define_binary_codes


class BinaryEncoderWrapper(BaseEncoder):
    def __init__(self, columns: list):
        self.columns = columns
        self.mappings = {}

    def fit(self, X: pd.DataFrame):
        for col in self.columns:
            if col in X.columns:
                self.mappings[col] = define_binary_codes(X[col])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns and col in self.mappings:
                X[col] = X[col].map(self.mappings[col])
        return X
