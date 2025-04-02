from typing import List

from preprocessing_kit.preprocessing.encoders.binary_encoder import BinaryEncoderWrapper
from preprocessing_kit.preprocessing.encoders.category_mapper import CategoryMapperWrapper
from preprocessing_kit.preprocessing.encoders.one_hot_encoder import OneHotEncoderWrapper
from preprocessing_kit.preprocessing.encoders.ordinal_encoder import OrdinalEncoderWrapper


def build_encoders_from_config(cfg) -> List:
    encoders = []

    if hasattr(cfg, "ordinal_encoder") and cfg.ordinal_encoder:
        encoders.append(OrdinalEncoderWrapper(cfg.ordinal_encoder))

    if hasattr(cfg, "one_hot_encoder") and cfg.one_hot_encoder:
        encoders.append(OneHotEncoderWrapper(cfg.one_hot_encoder))

    if hasattr(cfg, "binary_encoder") and cfg.binary_encoder:
        encoders.append(BinaryEncoderWrapper(cfg.binary_encoder))

    if hasattr(cfg, "category_mappings") and cfg.category_mappings:
        encoders.append(CategoryMapperWrapper(cfg.category_mappings))

    return encoders