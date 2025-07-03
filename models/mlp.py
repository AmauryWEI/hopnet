# File:         mlp.py
# Date:         2025/03/23
# Description:  Contains configurable standard Multi-Layer Perceptron (MLP)

from typing import Callable

import torch


def mlp(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    activation_func: Callable,
    num_layers: int = 2,
    final_activation: bool = True,
    layer_norm: bool = True,
) -> torch.nn.Sequential:
    assert num_layers >= 1
    layers = []

    # Special case when only a single linear layer is requested (direct in-out)
    if num_layers == 1:
        layers.append(torch.nn.Linear(in_channels, out_channels))
        if final_activation:
            layers.append(activation_func())
        return torch.nn.Sequential(*layers)

    layers = [torch.nn.Linear(in_channels, hidden_channels), activation_func()]
    for _ in range(1, num_layers - 1):
        layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
        layers.append(activation_func())

    # Create the last output layer
    layers.append(torch.nn.Linear(hidden_channels, out_channels))
    if final_activation:
        layers.append(activation_func())
    if layer_norm:
        layers.append(torch.nn.LayerNorm(out_channels))

    return torch.nn.Sequential(*layers)
