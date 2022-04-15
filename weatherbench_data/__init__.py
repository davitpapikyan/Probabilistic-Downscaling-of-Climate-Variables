"""Defines dataset and dataloader creation functionalities.
"""
import logging

import torch
from torch.nn.functional import interpolate
from torch.utils.data import Dataset, DataLoader

from .datasets import WeatherBenchData, TimeVariateData
from .datastorage import WeatherBenchNPYStorage
from .fileconverter import DATETIME_FORMAT
from .transforms import Transform
from .utils import log_dataset_info, prepare_datasets


def create_datasets(dataroot: str, name: str, train_min_date: str, train_max_date: str,
                    val_min_date: str, val_max_date: str, variables: list, transformation: Transform,
                    storage_root: str, apply_tranform_monthly: bool = True):
    """Creates transformed datasets.

    Args:
        dataroot: Path to the dataset.
        name: The name of the dataset.
        train_min_date: Minimum date starting from which to read the data for training.
        train_max_date: Maximum date until which to read the date for training.
        val_min_date: Minimum date starting from which to read the data for validation.
        val_max_date: Maximum date until which to read the date for validation.
        variables: variables: A list of WeatherBench variables.
        transformation: A transformation to fit.
        storage_root: A path to save metadata and fitted transformations.
        apply_tranform_monthly: Whether to apply transformation monthly or on the whole dataset.

    Returns:
        Training and validation datasets (already transformed), metadata containing longitude and
        latitude information for LR/HR data and monthly fitted transformations for eahc variable.
    """
    train_dataset, val_dataset, metadata, transformations = prepare_datasets(variables, train_min_date,
                                                                             train_max_date, val_min_date,
                                                                             val_max_date, dataroot,
                                                                             transformation, storage_root,
                                                                             apply_tranform_monthly)
    logger = logging.getLogger("base")
    log_dataset_info(train_dataset, f"Train {name}", logger)
    log_dataset_info(val_dataset, f"Validation {name}", logger)
    logger.info("Finished.\n")
    return train_dataset, val_dataset, metadata, transformations


def collate_wb_batch(samples: list):
    """Processes a list of samples to form a batch.

    Args:
        samples: A list of data points.

    Returns:
        A dictionary of the following items:
            LR – a low-resolution tensor,
            HR – a high-resolution tensor,
            INTERPOLATED – an upsampled low-resolution tensor with bicubic interpolation
        and a list of month indices corresponding to each sample.
    """
    lr_tensors, hr_tensors, months = [], [], []
    for lr, hr in samples:
        lr_tensors.append(torch.cat([variable[0] for variable in lr], dim=1))
        hr_tensors.append(torch.cat([variable[0] for variable in hr], dim=1))
        months.append(lr[0][2])
    fake_tensors = [interpolate(tensor, scale_factor=4, mode="bicubic") for tensor in lr_tensors]
    return {"LR": torch.cat(lr_tensors),
            "HR": torch.cat(hr_tensors),
            "INTERPOLATED": torch.cat(fake_tensors)}, months


def create_dataloaders(train_dataset: Dataset, val_dataset: Dataset, batch_size: int,
                       use_shuffle: bool = True, num_workers: int = None):
    """Creates train/val dataloaders.

    Args:
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        batch_size: The size of a batch.
        use_shuffle: Either shuffle the training data or not.
        num_workers: The number of processes for multi-process data loading.

    Returns:
        Training and validations dataloaders.
    """
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=collate_wb_batch,
                              shuffle=use_shuffle,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=num_workers)

    validation_loader = DataLoader(val_dataset,
                                   batch_size=32,
                                   collate_fn=collate_wb_batch,
                                   pin_memory=True,
                                   drop_last=True,
                                   num_workers=num_workers)

    return train_loader, validation_loader
