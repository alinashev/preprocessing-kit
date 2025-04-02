import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from configuration.configuration import Configuration
from preprocessing_kit.preprocessing.encoder_factory import build_encoders_from_config
from preprocessing_kit.preprocessing.encoders.scaler_wrapper import ScalerWrapper
from preprocessing_kit.preprocessing.encoders.target_encoder import TargetEncoderWrapper
from preprocessing_kit.preprocessing.processor_builder import PreprocessingPipelineBuilder
from preprocessing_kit.preprocessing.utils.utils import define_numerical_cols


class Processor:
    def __init__(
        self,
        target_col: str,
        unnecessary_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        scaler_numeric: bool = True,
        scaling_method: str = "standard",
        positive_target_label: str = "yes",
        config: Optional[Configuration] = None
    ):
        self.target_col = target_col
        self.unnecessary_columns = unnecessary_columns if unnecessary_columns else []
        self.test_size = test_size
        self.random_state = random_state
        self.scaler_numeric = scaler_numeric
        self.scaling_method = scaling_method
        self.positive_target_label = positive_target_label
        self.config = config

        self.input_cols: List[str] = []

    def _init_scaler(self, numeric_cols: List[str]) -> ScalerWrapper:
        if self.scaling_method == "minmax":
            return ScalerWrapper(MinMaxScaler(), numeric_cols)
        else:
            return ScalerWrapper(StandardScaler(), numeric_cols)

    def process(
        self, df: pd.DataFrame, split: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
        builder = PreprocessingPipelineBuilder(self, df)

        # Register encoders from config
        for encoder in build_encoders_from_config(self.config.encoding_config):
            builder.register_encoder(encoder)

        # Register scaler if needed
        if self.scaler_numeric:
            numeric_cols = define_numerical_cols(df)
            builder.register_encoder(self._init_scaler(numeric_cols))

        # Register target encoder
        builder.target_encoder = TargetEncoderWrapper({
            self.positive_target_label: 1
        })

        builder = (
            builder
            .with_column_removal()
            .with_encoding()
            .with_target_encoding()
        )
        builder = builder.with_split() if split else builder.without_split()

        return builder.build()
