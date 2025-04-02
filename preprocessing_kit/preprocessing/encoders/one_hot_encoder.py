from typing import List

import pandas as pd

from preprocessing_kit.preprocessing.encoders.base_encoder import BaseEncoder
from preprocessing_kit.preprocessing.utils.encoders import one_hot_encode


class OneHotEncoderWrapper(BaseEncoder):
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.encoder = None
        self.encoded_cols = []

    def fit(self, X: pd.DataFrame):
        X_encoded, self.encoder, self.encoded_cols = one_hot_encode(X.copy(), self.columns)
        self.X_fitted = X_encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.encoded_cols] = self.encoder.transform(X[self.columns].astype(str))
        X = X.drop(columns=self.columns)
        return X