from bci.data.base import EpochsDataset, available, load, register

# Importing the loader modules registers them.
from bci.data import synthetic  # noqa: F401

__all__ = ["EpochsDataset", "available", "load", "register"]
