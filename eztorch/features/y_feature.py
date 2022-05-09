import numpy as np
import pandas as pd
from .base_feature import BaseFeature


class YFeature(BaseFeature):

    @property
    def output_dim(self):
        return 1

    def extract_and_full_preprocess(self, full_dataset: pd.DataFrame) -> None:
        self.dataset = full_dataset[self.col_name].values
        preprocessed_dataset = self.preprocess_batch(self.dataset)
        if self.keep_preprocessed:
            self.preprocessed_dataset = preprocessed_dataset

    def preprocess_batch(self, batch: np.array) -> np.array:
        return batch
