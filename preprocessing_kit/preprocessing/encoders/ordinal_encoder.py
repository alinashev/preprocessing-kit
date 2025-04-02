from typing import List, Dict

import pandas as pd

from preprocessing_kit.preprocessing.encoders.base_encoder import BaseEncoder
from preprocessing_kit.preprocessing.utils.encoders import ordinal_encode


class OrdinalEncoderWrapper(BaseEncoder):
    def __init__(self, column_category_map: Dict[str, List[str]]):
        self.column_category_map = column_category_map
        self.encoder_map = {}

    def fit(self, X: pd.DataFrame):
        for col, categories in self.column_category_map.items():
            encoder = ordinal_encode(X[col], categories)
            self.encoder_map[col] = encoder

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.column_category_map:
            encoder = self.encoder_map[col]
            X[col] = encoder.transform(X[[col]].astype(str))
        return X
