import pandas as pd

from preprocessing_kit.preprocessing.data_processor import DataProcessor
from preprocessing_kit.preprocessing.processor import Processor


def main():
    df_init = pd.read_csv("../data/bank-additional-full.csv", sep=';')
    df = df_init.copy()
    processor = Processor(target_col="y", unnecessary_columns=["duration"], scaling_method="standard")
    data_processor = DataProcessor(
        processor=processor,
        apply_outliers_processing=True,
        apply_feature_engineering=True
    )

    data_processor.process(df)


if __name__ == "__main__":
    main()
