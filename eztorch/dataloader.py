from typing import List
import torch
import numpy as np

from features.base_feature import BaseFeature
from features.y_feature import YFeature


class DataFrameLoader(torch.utils.data.Dataset):
    """
        Dataset class that handles the data indexing for us and does the train/test split
        Call DataFrameLoader.set_test() to toggle it to use the validation dataset
        Call DataFrameLoader.set_train() to toggle it to use the train dataset
    """
    def __init__(
            self,
            x_features: List[BaseFeature],
            y_feature: YFeature,
            batch_size: int,
            train_test_split: float = 0.1) -> None:
        self.batch_size = batch_size
        self.train_batch_idxs = list(range(0, int(len(x_features[0].dataset) * (1 - train_test_split)), batch_size))
        self.test_batch_idxs = list(range(max(self.train_batch_idxs) + batch_size, len(x_features[0].dataset), batch_size))
        self.X = x_features
        self.Y = y_feature
        self.test = False

    def set_test(self):
        self.test = True

    def set_train(self):
        self.test = False

    def shuffle_idxs(self) -> None:
        np.random.shuffle(self.train_batch_idxs)

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, idx: int) -> List[List[torch.Tensor]]:
        if self.test:
            batch_start_idx = self.test_batch_idxs[idx]
        else:
            batch_start_idx = self.train_batch_idxs[idx]
        if batch_start_idx + self.batch_size >= len(self.X[0].dataset):
            raise StopIteration
        X = [torch.from_numpy(x[batch_start_idx:batch_start_idx+self.batch_size]) for x in self.X]
        Y = torch.from_numpy(self.Y[batch_start_idx:batch_start_idx+self.batch_size])
        return [X, Y]
