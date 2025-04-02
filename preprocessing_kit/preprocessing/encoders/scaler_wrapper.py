import pandas as pd
from typing import List
from .base_encoder import BaseEncoder


class ScalerWrapper(BaseEncoder):
    def __init__(self, scaler, numeric_cols: List[str]):
        self.scaler = scaler
        self.numeric_cols = numeric_cols

    def fit(self, X: pd.DataFrame):
        self.scaler.fit(X[self.numeric_cols])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        return X
