from dataclasses import dataclass
from typing import Optional, Dict, List, Any


@dataclass
class EncoderConfig:
    ordinal_encoder: Optional[Dict[str, List[str]]] = None
    one_hot_encoder: Optional[List[str]] = None
    category_mappings: Optional[Dict[str, Dict[str, int]]] = None
    binary_encoder: Optional[List[str]] = None


@dataclass
class Configuration:
    encoding_config: EncoderConfig = None
    outlier_config: Dict[str, Any] = None
    missing_values: Dict[str, Any] = None
    feature_engineering: Dict[str, Any] = None
