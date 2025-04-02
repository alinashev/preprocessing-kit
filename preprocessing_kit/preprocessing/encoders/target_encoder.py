import pandas as pd
from .base_encoder import BaseEncoder

class TargetEncoderWrapper:
    def __init__(self, mapping: dict):
        self.mapping = mapping

    def encode(self, y: pd.Series, y_val: pd.Series = None):
        if y_val is not None:
            return y.map(self.mapping), y_val.map(self.mapping)
        else:
            return y.map(self.mapping)


