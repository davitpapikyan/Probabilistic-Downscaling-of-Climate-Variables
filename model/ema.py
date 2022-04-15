"""Defines Exponential Moving Average class for
model parameters.

The work is based on https://github.com/ermongroup/ddim/blob/main/models/ema.py.
"""

import torch.nn as nn


class EMA(object):
    """An Exponential Moving Average class.

    Attributes:
        mu: IDs of gpus.
        shadow: The storage for parameter values.
    """

    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        """Registers network parameters.

        Args:
            module: A parameter module, typically a neural network.
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        """Updates parameters with a decay rate mu and stores in a storage.

        Args:
            module: A parameter module, typically a neural network.
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        """Updates network parameters from the storage.

        Args:
            module: A parameter module, typically a neural network.
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        """Updates network parameters from the storage and returns a copy of it.

        Args:
            module: A parameter module, typically a neural network.

        Returns:
            A copy of network parameters.
        """
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(
                inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())

        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        """Returns current state of model parameters.

        Returns:
            Current state of model parameters stored in a local storage.
        """
        return self.shadow

    def load_state_dict(self, state_dict):
        """Update local storage of parameters.

        Args:
            state_dict: A state of network parameters for updating local storage.
        """
        self.shadow = state_dict
