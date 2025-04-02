import pandas as pd
from typing import Callable, List, Optional, Tuple

from preprocessing_kit.preprocessing.encoders.target_encoder import TargetEncoderWrapper
from preprocessing_kit.preprocessing.utils.utils import remove_unnecessary_columns, split_data


class PreprocessingPipelineBuilder:
    def __init__(self, pipeline, df: pd.DataFrame):
        self.pipeline = pipeline
        self.df = df
        self.steps: List[Callable] = []
        self.split = True
        self.config = pipeline.config
        self.target_col = pipeline.target_col
        self.test_size = pipeline.test_size
        self.random_state = pipeline.random_state

        self.X_train: Optional[pd.DataFrame] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_val: Optional[pd.Series] = None

        self.encoders: List = []
        self.target_encoder: Optional[TargetEncoderWrapper] = None

    def register_encoder(self, encoder):
        self.encoders.append(encoder)
        return self

    def with_column_removal(self):
        def step():
            self.df = remove_unnecessary_columns(self.df, self.pipeline.unnecessary_columns)
        self.steps.append(step)
        return self

    def with_split(self):
        self.split = True
        return self

    def without_split(self):
        self.split = False
        return self

    def with_encoding(self):
        def step():
            for encoder in self.encoders:
                if self.split:
                    self.X_train, self.X_val = encoder.transform_pair(self.X_train, self.X_val)
                else:
                    self.X_train = encoder.fit_transform(self.X_train)
        self.steps.append(step)
        return self

    def with_target_encoding(self):
        def step():
            if self.target_encoder:
                if self.split:
                    self.y_train, self.y_val = self.target_encoder.encode(self.y_train, self.y_val)
                else:
                    self.y_train = self.target_encoder.encode(self.y_train)
        self.steps.append(step)
        return self

    def build(self) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = split_data(
                self.df, self.target_col, self.test_size, self.random_state
            )
        else:
            self.X_train = self.df.drop(columns=[self.target_col])
            self.y_train = self.df[self.target_col]
            self.X_val = None
            self.y_val = None

        for step in self.steps:
            step()

        self.pipeline.input_cols = list(self.X_train.columns)

        return self.X_train, self.y_train, self.X_val, self.y_val
