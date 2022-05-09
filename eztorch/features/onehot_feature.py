import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .base_feature import BaseFeature


class OneHotFeature(BaseFeature):
    _encoder = None

    @property
    def output_dim(self):
        return len(self._encoder.categories_[0])

    def extract_and_full_preprocess(self, full_dataset: pd.DataFrame) -> None:
        self._encoder = OneHotEncoder()
        self.dataset = full_dataset[self.col_name].values
        if self.keep_preprocessed:
            self.preprocessed_dataset = self._encoder.fit_transform(self.dataset[:, np.newaxis]).toarray()
        else:
            self._encoder.fit(self.dataset[:, np.newaxis]).toarray()

    def preprocess_batch(self, batch: np.array) -> np.array:
        return self._one_hot_encode(batch)

    def _one_hot_encode(self, batch: np.array) -> np.array:
        return self._encoder.transform(batch[:, np.newaxis]).toarray()
