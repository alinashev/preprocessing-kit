from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple

class BaseEncoder(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

    def transform_pair(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.fit(X_train)
        return self.transform(X_train), self.transform(X_val)


