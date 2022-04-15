"""Defines auxiliary functionalities for data transformation."""
import os
import pickle
from collections import OrderedDict
from datetime import datetime
from types import SimpleNamespace

import torch
from dateutil.relativedelta import relativedelta
from torch.utils.data import Dataset, ConcatDataset

from .datasets import WeatherBenchData, TimeVariateData
from .datastorage import WeatherBenchNPYStorage
from .fileconverter import DATETIME_FORMAT
from .transforms import Transform

ONE_MONTH = relativedelta(months=1)


def reverse_transform_variable(transformations: dict, variable: str, data_type: str, months: list,
                               tensor: torch.tensor, apply_tranform_monthly: bool) -> torch.tensor:
    """Inverse transforms a single variable.

    Args:
        transformations: A dictionary of monthly fitted LR/HR transformations for each variable used for training.
        variable: Variable name from WeatherBench dataset.
        data_type: Either lr or hr.
        months: A list of month indices.
        tensor: Tensor data of shape (batch_size, 1, H, W).
        apply_tranform_monthly: Whether to apply transformation monthly or on the whole dataset.

    Returns:
        Returns inverse transformed tensor by preserving the dimensionality.
    """
    batch_size = tensor.shape[0]
    if apply_tranform_monthly:
        return torch.cat([transformations[variable][data_type][months[idx]].revert(tensor[idx])
                          for idx in range(batch_size)])
    else:
        return torch.cat([transformations[variable][data_type][0].revert(tensor[idx])
                          for idx in range(batch_size)])


def reverse_transform_tensor(tensor: torch.tensor, transformations: dict,
                             variables: list, data_type: str, months: list,
                             apply_tranform_monthly: bool) -> torch.tensor:
    """Inverse transforms tensor.

    Args:
        tensor: Tensor data of shape (batch_size, number of variables, H, W).
        transformations: A dictionary of monthly fitted LR/HR transformations for each variable used for training.
        variables: A list of WeatherBench variables.
        data_type: Either lr or hr.
        months: A list of month indices.
        apply_tranform_monthly: Whether to apply transformation monthly or on the whole dataset.

    Returns:
        Inverse transformed single data point.
    """
    reverse_transformed_tesnors = []
    for index, variable in enumerate(variables):
        tensor_of_variable = tensor[:, index].unsqueeze(1)
        reverse_transformed_tesnors.append(reverse_transform_variable(transformations, variable,
                                                                      data_type, months,
                                                                      tensor_of_variable,
                                                                      apply_tranform_monthly))
    return torch.cat(reverse_transformed_tesnors, dim=1)


def reverse_transform(data: dict, transformations: dict,
                      variables: list, months: list, apply_tranform_monthly: bool) -> dict:
    """Inverse transforms data stored in a dictionary.

    Args:
        data: Dictionary of data points.
        transformations: A dictionary of monthly fitted LR/HR transformations for each variable used for training.
        variables: A list of WeatherBench variables.
        months: A list of month indices.
        apply_tranform_monthly: Whether to apply transformation monthly or on the whole dataset.

    Returns:
        Inverse transformed data.
    """
    reverse_transformed_batch = OrderedDict({})
    for key, tensor in data.items():
        if key == "LR":
            reverse_transformed_batch[key] = reverse_transform_tensor(tensor, transformations, variables,
                                                                      "lr", months, apply_tranform_monthly)
        else:
            reverse_transformed_batch[key] = reverse_transform_tensor(tensor, transformations, variables,
                                                                      "hr", months, apply_tranform_monthly)
    return reverse_transformed_batch


def get_start_of_next_month(datetime_object: datetime) -> datetime:
    """Computes the start of the next month of the datetime_object.

    Args:
        datetime_object: A datetime object representing a particular datetime.

    Returns:
        The start date of the next month of datetime_object.
    """
    return (datetime_object + ONE_MONTH).replace(day=1)


def get_str_date(datetime_object: datetime) -> str:
    """Converts datetime object into string.

    Args:
        datetime_object: A datetime object representing a particular datetime.

    Returns:
        Returns datetime_object converted into string according to DATETIME_FORMAT.
    """
    return datetime.strftime(datetime_object, DATETIME_FORMAT)


def read_variable(dataroot: str, data_type: str, variable: str, min_date: str, max_date: str) -> TimeVariateData:
    """Reads a single variable data.

    Args:
        dataroot: Path to the dataset.
        data_type: Either lr or hr.
        variable: Variable name from WeatherBench dataset.
        min_date: Minimum date starting from which to read the data.
        max_date: Maximum date until which to read the date.

    Returns:
        TimeVariateData of variable.
    """
    return TimeVariateData(WeatherBenchNPYStorage(os.path.join(dataroot, data_type, variable)),
                           name=f"{variable}_{data_type}{min_date}",
                           lead_time=0, min_date=min_date, max_date=max_date)


def add_monthly_data(storage: dict, new_data: TimeVariateData, month: int) -> None:
    """Adds new_data to storage with a key month.

    Args:
        storage: A dictionary to add monthly data. Keys are indices of months.
        new_data: Data to add.
        month: To which month the data belongs.
    """
    storage[month] = ConcatDataset([storage[month], new_data]) if month in storage else new_data


def create_global_dataset(min_date: str, max_date: str, dataroot: str, data_type: str, variable: str) -> dict:
    """Reads data entirely and constructs a dictionary mapping 0 to the dataset.

    Args:
        min_date: Minimum date starting from which to read the data.
        max_date: Maximum date until which to read the date.
        dataroot: Path to the dataset.
        data_type: Either lr or hr.
        variable: Variable name from WeatherBench dataset.

    Returns:
        Dictionary mapping 0 to the dataset.
    """
    return {0: read_variable(dataroot=dataroot, data_type=data_type,
                             variable=variable, min_date=min_date,
                             max_date=max_date)}


def create_monthly_datasets(min_date: str, max_date: str, dataroot: str, data_type: str, variable: str) -> dict:
    """Reads data month by month and concatenates datasets of the same month. Constructs
    a dictionary mapping each month index to its corresponding dataset.

    Args:
        min_date: Minimum date starting from which to read the data.
        max_date: Maximum date until which to read the date.
        dataroot: Path to the dataset.
        data_type: Either lr or hr.
        variable: Variable name from WeatherBench dataset.

    Returns:
        Month to data mapping.
    """
    month2data = {}
    max_date_datetime = datetime.strptime(max_date, DATETIME_FORMAT)
    start = datetime.strptime(min_date, DATETIME_FORMAT)
    start_of_next_month = start + ONE_MONTH

    while start_of_next_month < max_date_datetime:
        current_month = start.month
        data = read_variable(dataroot=dataroot, data_type=data_type,
                             variable=variable,
                             min_date=get_str_date(start),
                             max_date=get_str_date(start_of_next_month))
        add_monthly_data(month2data, data, current_month)
        start = start_of_next_month
        start_of_next_month = get_start_of_next_month(start_of_next_month)

    data = read_variable(dataroot=dataroot, data_type=data_type,
                         variable=variable,
                         min_date=get_str_date(start),
                         max_date=get_str_date(max_date_datetime))
    add_monthly_data(month2data, data, start.month)

    if not all(month in month2data for month in range(1, 13)):
        month2data[0] = ConcatDataset([data for data in month2data.values()])

    return month2data


def unpack_datasets(datasets) -> list:
    """Unpacks a concatenated datasets and creates alist of those datasets.

    Args:
        datasets: Either ConcatDataset object or TimeVariateData.

    Returns:
        A list of TimeVariateData datasets. If datasets is TimeVariateData,
        object, returns that object NOT in a list.
    """
    return [unpack_datasets(dataset) for dataset in datasets.datasets] \
        if isinstance(datasets, ConcatDataset) else datasets


def flatten(list_of_lists):
    """Flattens a nested-list structure.

    Args:
        list_of_lists: A list of nested lists.

    Returns:
        Flattened 1-dimensional list.
    """
    if not isinstance(list_of_lists, list):
        return [list_of_lists]
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def fit_monthly_transformations(datasets: list, transformation: Transform) -> Transform:
    """Fits a transformation to a list of datasets.

    Args:
        datasets: A list of datasets corresponding to the same month.
        transformation: A transformation to fit.

    Returns:
        A fitted transformation.
    """
    transform = transformation()
    for data in flatten(unpack_datasets(datasets)):
        transform.fit(data)
        transform.clear_data_source()
    return transform


def store_monthly_transformations(data: dict, transformation: Transform) -> dict:
    """Creates a month to transformation mapping.

    Args:
        data: A dictionary of datasets of each month. Keys are indices of months.
        transformation: A transformation to fit.

    Returns:
        A dictionary containing fitted transformation for each monthly data.
    """
    return {month: fit_monthly_transformations(datasets, transformation) for month, datasets in data.items()}


def fit_and_return_transformations(min_date: str, max_date: str, dataroot: str, data_type: str,
                                   variable: str, transformation: Transform,
                                   apply_tranform_monthly: bool = True) -> dict:
    """Creates monthly transformations.

    Args:
        min_date: Minimum date starting from which to read the data.
        max_date: Maximum date until which to read the date.
        dataroot: Path to the dataset.
        data_type: Either lr or hr.
        variable: Variable name from WeatherBench dataset.
        transformation: A transformation to fit.
        apply_tranform_monthly: Whether to apply transformation monthly or on the whole dataset.

    Returns:
        A dictionary mapping each month to its fitted transformaition.
    """

    if apply_tranform_monthly:
        data = create_monthly_datasets(min_date, max_date, dataroot, data_type, variable)
    else:
        data = create_global_dataset(min_date, max_date, dataroot, data_type, variable)
    return store_monthly_transformations(data, transformation)


def save_object(obj, path: str, filename: str) -> None:
    """Saves python object with pickle.

    Args:
        obj: Object to save.
        path: A directory where to save.
        filename: The name of a file in which to write.
    """
    if not filename.endswith(".pkl"):
        filename = f"{filename}.pkl"

    with open(os.path.join(path, filename), "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def load_object(path: str, filename: str):
    """Loads python object.

    Args:
        path: A directory where an object is saved.
        filename: The name of a file in which an object is saved.

    Returns:
        Loaded object if path and filename are correct, otherwise None.
    """
    try:
        if not filename.endswith(".pkl"):
            filename = f"{filename}.pkl"

        with open(os.path.join(path, filename), "rb") as file:
            return pickle.load(file)

    except FileNotFoundError:
        return None


def prepare_datasets(variables: list, train_min_date: str, train_max_date: str, val_min_date: str, val_max_date: str,
                     dataroot: str, transformation: Transform, storage_root: str = None,
                     apply_tranform_monthly: bool = True):
    """Reads datasets and fits trasformation.

    Args:
        variables: A list of WeatherBench variables.
        train_min_date: Minimum date starting from which to read the data for training.
        train_max_date: Maximum date until which to read the date for training.
        val_min_date: Minimum date starting from which to read the data for validation.
        val_max_date: Maximum date until which to read the date for validation.
        dataroot: Path to the dataset.
        transformation: A transformation to fit.
        storage_root: A path to save metadata and fitted transformations.
        apply_tranform_monthly: Whether to apply transformation monthly or on the whole dataset.

    Returns:
        Training and validation datasets, metadata and fitted transformations.
    """
    train_datasets, val_datasets = {"lr": [], "hr": []}, {"lr": [], "hr": []}
    transformations, metadata = {}, {}

    for idx, variable in enumerate(variables):

        transformations[variable] = {}
        for data_type in ("lr", "hr"):
            month2transform = fit_and_return_transformations(train_min_date, train_max_date, dataroot,
                                                             data_type, variable, transformation,
                                                             apply_tranform_monthly)
            wbd_storage = WeatherBenchNPYStorage(os.path.join(dataroot, data_type, variable))
            train_data = TimeVariateData(wbd_storage, name=f"train_{data_type}_{variable}",
                                         lead_time=0, min_date=train_min_date,
                                         max_date=train_max_date, transform=month2transform)
            # Updates metadata information for only first variable, other variables should have
            # the same latitudes and longitudes.
            if idx == 0:
                metadata.update({f"{data_type}_{dimension['name']}": dimension["values"]
                                 for dimension in wbd_storage.meta_data["coords"]})
            transformations[variable][f"{data_type}"] = month2transform
            train_datasets[data_type].append(train_data)
            val_data = TimeVariateData(WeatherBenchNPYStorage(os.path.join(dataroot, data_type, variable)),
                                       name=f"{data_type}_{variable}", lead_time=0,
                                       min_date=val_min_date, max_date=val_max_date,
                                       transform=month2transform)
            val_datasets[data_type].append(val_data)

    train_dataset = WeatherBenchData(min_date=train_min_date, max_date=train_max_date)
    train_dataset.add_data_group("lr", train_datasets["lr"])
    train_dataset.add_data_group("hr", train_datasets["hr"])

    val_dataset = WeatherBenchData(min_date=val_min_date, max_date=val_max_date)
    val_dataset.add_data_group("lr", val_datasets["lr"])
    val_dataset.add_data_group("hr", val_datasets["hr"])

    metadata = SimpleNamespace(**metadata)

    if storage_root:
        save_object(metadata, storage_root, "metadata")
        save_object(transformations, storage_root, "transformations")

    return train_dataset, val_dataset, metadata, transformations


def prepare_test_data(variables: list, val_min_date: str, val_max_date: str,
                      dataroot: str, transformations: dict):
    """Creates testing data with already fitted transformations.

    Args:
        variables: A list of WeatherBench variables.
        val_min_date: Minimum date starting from which to read the data for validation.
        val_max_date: Maximum date until which to read the date for validation.
        dataroot: Path to the dataset.
        transformations: A dict of month to transformation mappings.

    Returns:
        Test data.
    """
    val_datasets = {"lr": [], "hr": []}

    for idx, variable in enumerate(variables):
        for data_type in ("lr", "hr"):
            val_data = TimeVariateData(WeatherBenchNPYStorage(os.path.join(dataroot, data_type, variable)),
                                       name=f"{data_type}_{variable}", lead_time=0,
                                       min_date=val_min_date, max_date=val_max_date,
                                       transform=transformations[variable][data_type])
            val_datasets[data_type].append(val_data)

    val_dataset = WeatherBenchData(min_date=val_min_date, max_date=val_max_date)
    val_dataset.add_data_group("lr", val_datasets["lr"])
    val_dataset.add_data_group("hr", val_datasets["hr"])

    return val_dataset


def log_dataset_info(dataset: Dataset, dataset_name: str, logger) -> None:
    """Logs dataset information.

    Args:
        dataset: A pytorch dataset.
        dataset_name: The name of dataset.
        logger: Logging object.
    """
    logger.info(f"Dataset [{dataset.__class__.__name__} - {dataset_name}] is created.")
    logger.info(f"""Created {dataset.__class__.__name__} dataset of length {len(dataset)}, containing data 
    from {dataset.min_date} until {dataset.max_date}""")
    logger.info(f"Group structure: {dataset.get_data_names()}")
    logger.info(f"Channel count: {dataset.get_channel_count()}\n")
