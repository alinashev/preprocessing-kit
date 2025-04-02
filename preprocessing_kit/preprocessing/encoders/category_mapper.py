import pandas as pd
from .base_encoder import BaseEncoder

class CategoryMapperWrapper(BaseEncoder):
    def __init__(self, mappings: dict):
        self.mappings = mappings

    def fit(self, X: pd.DataFrame):
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self.mappings.items():
            X[col] = X[col].map(mapping)
        return X


