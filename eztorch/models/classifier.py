from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from eztorch.features.base_feature import BaseFeature
from eztorch.features.embedded_feature import EmbeddedFeature
from eztorch.features.onehot_feature import OneHotFeature
from eztorch.features.numerical_feature import NumericalFeature


class ClassifierModel(torch.nn.Module):
    """
        Simple feedforward classifier model (single or multi-class)
    """

    def __init__(
            self,
            features: List[BaseFeature],
            hidden_sizes: Tuple[int, ...],
            batch_size: int,
            batch_norm: bool = False,
            dropout: float = 0,
            activation=None,
            num_classes: int = 1) -> None:
        super().__init__()
        self.input_layers = nn.ModuleList([])
        self.activation = F.relu if activation is None else activation
        self.num_classes = num_classes

        concat_layer_size = 0
        for feature in features:
            if isinstance(feature, EmbeddedFeature):
                input_layer = torch.nn.Embedding(feature.num_embeddings, feature.output_dim)
                concat_layer_size += feature.output_dim
            elif isinstance(feature, OneHotFeature):
                input_layer = torch.nn.Identity(feature.output_dim)
                concat_layer_size += feature.output_dim
            elif isinstance(feature, NumericalFeature):
                input_layer = torch.nn.Identity()
                concat_layer_size += 1
            else:
                raise RuntimeError(f"Unexpected feature {feature}!")
            self.input_layers.append(input_layer)

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(batch_size)
        else:
            self.batch_norm = None

        input_size = concat_layer_size
        self.hidden_layers = nn.ModuleList([])
        for size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(input_size, size))
            input_size = size

        self.output_layer = nn.Linear(input_size, self.num_classes)

    def forward(self, X):
        for i in range(len(X)):
            X[i] = self.input_layers[i](X[i])
            if len(X[i].shape) == 2:
                X[i] = X[i].reshape([1, X[i].shape[0], X[i].shape[1]])
            else:
                X[i] = X[i].reshape([1, X[i].shape[0], 1])
        X = torch.concat(X, dim=2).float()
        if self.batch_norm is not None:
            X = self.batch_norm(X)
        if self.dropout is not None:
            X = self.dropout(X)
        for hidden in self.hidden_layers:
            X = self.activation(hidden(X))
        if self.num_classes > 1:
            return torch.softmax(self.output_layer(X), self.num_classes)
        return torch.sigmoid(self.output_layer(X))
