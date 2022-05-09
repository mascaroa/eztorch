import numpy as np
from scipy import stats
from base_feature import BaseFeature


class NumericalFeature(BaseFeature):
    _max = None
    _min = None

    @property
    def output_dim(self):
        return 1

    def extract_and_full_preprocess(self, full_dataset: pd.DataFrame) -> None:
        self.dataset = full_dataset[self.col_name].values
        preprocessed_dataset = self.preprocess_batch(self.dataset)
        if self.keep_preprocessed:
            self.preprocessed_dataset = preprocessed_dataset

    def preprocess_batch(self, batch: np.array) -> np.array:
        if self.params.get("norm") == "minmax":  # Min-max normalization
            if self._max is None:
                self._max = batch.max()
            if self._min is None:
                self._min = batch.min()
            batch = (batch - self._min) / (self._max - self._min)
        elif self.params.get("norm") == "boxcox":
                batch, self.bclambda = stats.boxcox(batch - self.dataset.min() + 1e-4)
        return batch
