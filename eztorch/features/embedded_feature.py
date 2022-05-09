import numpy as np
import pandas as pd

from eztorch.util import Map
from .base_feature import BaseFeature


class EmbeddedFeature(BaseFeature):
    _map = None

    @property
    def num_embeddings(self):
        return len(self._map)

    @property
    def output_dim(self):
        return self.params.get("embedding_dim")

    def extract_and_full_preprocess(self, full_dataset: pd.DataFrame) -> None:
        self._map = Map()
        self.dataset = full_dataset[self.col_name].values
        preprocessed_dataset = [self._map.add(item) for item in self.dataset]
        if self.keep_preprocessed:
            self.preprocessed_dataset = preprocessed_dataset

    def preprocess_batch(self, batch: np.array) -> np.array:
        return self._embed(batch)

    def _embed(self, batch: np.array) -> np.array:
        batch = [self._map[item] for item in batch]
        return np.array(batch)
