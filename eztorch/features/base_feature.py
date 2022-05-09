from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from eztorch.util import lower_case_dict_keys


class BaseFeature(ABC):
    """
        Abstract base class for all Features
        Implements the init common to all Features that takes in a column_name to link the feature to the dataset
    """

    ID = 0
    dataset = None
    preprocessed_dataset = None

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.ID = cls.ID
        cls.ID += 1
        return obj

    def __init__(self, column_name: str, params: Optional[dict] = None) -> None:
        self.col_name = column_name
        self.params = lower_case_dict_keys(params) if params is not None else {}
        # Parameter to keep the preprocessed dataset (instead of doing it on the fly)
        self.keep_preprocessed = self.params.get("keep_preprocessed", True)

    @abstractmethod
    def extract_and_full_preprocess(self, full_dataset: pd.DataFrame) -> None:
        """
            To be called with the entire dataset before model training
            in order to generate mappings or calculate any constants that
            require the entire dataset
        """
        pass

    @abstractmethod
    def preprocess_batch(self, batch: np.array) -> np.array:
        """
            Called with each batch of data to do any preprocessing
            if necessary.
        """
        pass

    def __len__(self):
        if self.dataset is not None:
            return len(self.dataset)
        return 0

    def __getitem__(self, idx: slice) -> np.array:
        # This isn't my favourite way of doing this, normally I would try to decouple the datasource
        if self.preprocessed_dataset is not None:
            return self.preprocessed_dataset[idx]
        return self.preprocess_batch(self.dataset[idx])

    def __str__(self):
        return f"{self.col_name}_{self.ID}"

    def __repr__(self):
        return f"{self.__class__.__name__}[{self}]"
