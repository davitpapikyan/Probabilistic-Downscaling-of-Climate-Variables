"""Defines core classes to read WeatherBench data.
"""

from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset

from .datastorage import WeatherBenchNPYStorage
from .fileconverter import TEMPORAL_RESOLUTION, DATETIME_FORMAT


def _parse_date_input(date_input, datetime_format=None):
    if date_input is None:
        return None
    input_type = type(date_input)
    if input_type == np.datetime64:
        return date_input
    elif input_type == datetime:
        return np.datetime64(date_input)
    elif input_type == str:
        if datetime_format is None:
            datetime_format = DATETIME_FORMAT
        try:
            date = datetime.strptime(date_input, datetime_format)
        except Exception:
            raise Exception(
                "[ERROR] Encountered invalid date string input (input: {}, datetime format: {}).".format(
                    date_input, datetime_format
                )
            )
        return np.datetime64(date)
    else:
        raise Exception("[ERROR] Encountered invalid date input.")


def _verify_date_bounds(min_date, max_date):
    assert (isinstance(min_date, np.datetime64) or min_date is None) and (isinstance(max_date, np.datetime64) or max_date is None), \
        "[ERROR] Date bounds must be given as numpy.datetime64 objects."
    if min_date is not None:
        assert (min_date - np.datetime64("2020-01-01T00")) % TEMPORAL_RESOLUTION == np.timedelta64(0, "ms"), \
            "[ERROR] Date bounds must be consistent with the temporal resolution of the data set ({}).".format(
                TEMPORAL_RESOLUTION
            )
    if max_date is not None:
        assert (max_date - np.datetime64("2020-01-01T00")) % TEMPORAL_RESOLUTION == np.timedelta64(0, "ms"), \
            "[ERROR] Date bounds must be consistent with the temporal resolution of the data set ({}).".format(
                TEMPORAL_RESOLUTION
            )
    if min_date is not None and max_date is not None:
        assert max_date > min_date, "[ERROR] Lower date bound ({}) must be earlier than upper ({}).".format(min_date, max_date)


class DefaultIdentityMapping(dict):
    def __missing__(self, key):
        return lambda x: x


class TimeVariateData(Dataset):

    def __init__(self, source, name=None, lead_time=None, delays=None, min_date=None,
                 max_date=None, transform: dict = None):
        assert isinstance(source, WeatherBenchNPYStorage)
        assert source.is_time_variate()
        self.name = name if name is not None else source.name
        if name is not None:
            assert isinstance(name, str)
        self.source = source
        self._lead_time = TEMPORAL_RESOLUTION * lead_time if lead_time is not None else None
        if delays is not None:
            assert isinstance(delays, list), "[ERROR] Delay parameter must be given as list."
            for d in delays:
                assert isinstance(d, int), "[ERROR] Delay parameter must be given as list of ints."
            if 0 not in delays:
                delays = [0] + delays
            delays = np.array(delays)
            assert len(delays) == len(np.unique(delays)), "[ERROR] Delays must be unique."
            self._delays = TEMPORAL_RESOLUTION * delays
        else:
            self._delays = None
        self.min_date = None
        self.max_date = None
        self._sample_index = None
        self.set_date_range(min_date, max_date)
        self._fitting_mode = False
        self._transform = transform if transform else DefaultIdentityMapping()

    def set_date_range(self, min_date=None, max_date=None, datetime_format=None):
        min_date = _parse_date_input(min_date, datetime_format)
        max_date = _parse_date_input(max_date, datetime_format)
        _verify_date_bounds(min_date, max_date)
        source_time_stamps = self.source.get_valid_time_stamps()
        source_min_date = np.min(source_time_stamps)
        source_max_date = np.max(source_time_stamps) + TEMPORAL_RESOLUTION
        admissible_min_date = source_min_date
        admissible_max_date = source_max_date
        if self._lead_time is not None:
            admissible_min_date = admissible_min_date - self._lead_time
            admissible_max_date = admissible_max_date - self._lead_time
        if self._delays is not None:
            admissible_min_date = admissible_min_date - np.min(self._delays)
            admissible_max_date = admissible_max_date - np.max(self._delays)
        if min_date is None:
            self.min_date = admissible_min_date
        else:
            assert min_date >= admissible_min_date, \
                "[ERROR] Requested minimum date ({}) is beyond the range of admissible dates ([{}] – [{}]).".format(
                    min_date, admissible_min_date, admissible_max_date
                )
            self.min_date = min_date
        if max_date is None:
            self.max_date = admissible_max_date
        else:
            assert max_date <= admissible_max_date, \
                "[ERROR] Requested maximum date ({}) is beyond the range of admissible dates ([{}] – [{}]).".format(
                    max_date, admissible_min_date, admissible_max_date
                )
            self.max_date = max_date
        self._build_sample_index()
        return self

    def _build_sample_index(self):
        valid_samples = np.arange(self.min_date, self.max_date, TEMPORAL_RESOLUTION)
        self._sample_index = {i: time_stamp for i, time_stamp in enumerate(valid_samples)}

    def set_transform(self, transform: dict):
        self._transform = transform

    def __getitem__(self, item):
        time_stamp = self._sample_index[item]
        month = int(time_stamp.astype("datetime64[M]").astype(int) % 12 + 1)

        if month not in self._transform:
            month = 0

        if self._lead_time is not None:
            time_stamp = time_stamp + self._lead_time
        if self._fitting_mode or self._delays is None:
            return self._transform[month](self.source[time_stamp]), self.name, month
        else:
            return tuple((self._transform[month](self.source[delayed_time]), self.name, month)
                         for delayed_time in (time_stamp + self._delays))

    def __len__(self):
        return len(self._sample_index)

    def get_channel_count(self):
        source_channels = self.source.get_channel_count()
        if self._delays is not None:
            return len(self._delays) * source_channels
        else:
            return source_channels

    @staticmethod
    def _generate_batches(data, data_length: int, chunk_size: int = 50000):
        for start in range(0, data_length, chunk_size):
            yield [next(data) for _ in range(start, min(data_length, start+chunk_size))]

    def get_batch(self, indices):
        data = (self.__getitem__(i) for i in indices)
        for chunk_of_data in self._generate_batches(data, len(indices)):
            if self._delays is not None:
                yield tuple(torch.cat(d[0], dim=0) for d in chunk_of_data)
            else:
                yield torch.cat([tup[0] for tup in chunk_of_data], dim=0)

    def enable_fitting_mode(self):
        return self.set_fitting_mode(True)

    def disable_fitting_mode(self):
        return self.set_fitting_mode(False)

    def set_fitting_mode(self, mode):
        assert isinstance(mode, bool)
        self._fitting_mode = mode
        return self

    def get_fitting_mode(self):
        return self._fitting_mode

    @staticmethod
    def is_time_variate():
        return True

    def summarize(self):
        return {
            "data_type": "TimeVariateData",
            "path": self.source.path,
            "date_range": [
                self._numpy_date_to_datetime(self.min_date).strftime(DATETIME_FORMAT),
                self._numpy_date_to_datetime(self.max_date).strftime(DATETIME_FORMAT)
            ],
            "lead_time": self._lead_time,
            "delays": self._delays,
        }

    @staticmethod
    def _numpy_date_to_datetime(time_stamp):
        total_seconds = (time_stamp - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
        return datetime.utcfromtimestamp(total_seconds)

    def get_valid_time_stamps(self):
        return sorted(self._sample_index.values())


class ConstantData(Dataset):
    def __init__(self, source, name=None, min_date=None, max_date=None, datetime_format=None):
        assert isinstance(source, WeatherBenchNPYStorage)
        assert not source.is_time_variate()
        if name is not None:
            assert isinstance(name, str)
        self.name = name if name is not None else source.name
        self.source = source
        min_date = _parse_date_input(min_date, datetime_format)
        max_date = _parse_date_input(max_date, datetime_format)
        _verify_date_bounds(min_date, max_date)
        self.min_date = min_date
        self.max_date = max_date
        self._num_samples = None
        self.set_date_range(min_date, max_date)
        self._fitting_mode = False

    def __getitem__(self, item):
        if item < self._num_samples:
            return self.source[item]
        else:
            raise Exception("[ERROR] Requested item ({}) is beyond the range of valid item numbers ([0, {}]).".format(item, self._num_samples))

    def __len__(self):
        return self._num_samples

    def set_date_range(self, min_date=None, max_date=None, datetime_format=None):
        min_date = _parse_date_input(min_date, datetime_format)
        max_date = _parse_date_input(max_date, datetime_format)
        _verify_date_bounds(min_date, max_date)
        self.min_date = min_date
        self.max_date = max_date
        if min_date is None or max_date is None:
            self._num_samples = 1
        else:
            self._num_samples = int((max_date - min_date) / TEMPORAL_RESOLUTION)
        return self

    def get_channel_count(self):
        return self.source.get_channel_count()

    def enable_fitting_mode(self):
        return self.set_fitting_mode(True)

    def disable_fitting_mode(self):
        return self.set_fitting_mode(False)

    def set_fitting_mode(self, mode):
        assert isinstance(mode, bool)
        self._fitting_mode = mode
        return self

    def get_fitting_mode(self):
        return self._fitting_mode

    @staticmethod
    def is_time_variate():
        return False

    def summarize(self):
        return {
            "data_type": "ConstantData",
            "path": self.source.path
        }


class WeatherBenchData(Dataset):
    def __init__(self, min_date=None, max_date=None, datetime_format=None):
        min_date = _parse_date_input(min_date, datetime_format)
        max_date = _parse_date_input(max_date, datetime_format)
        _verify_date_bounds(min_date, max_date)
        self.min_date = min_date
        self.max_date = max_date
        self.data_groups = OrderedDict({})

    def add_data_group(self, group_key, datasets, _except_on_changing_date_bounds=False):
        self._verify_data_group_inputs(group_key, datasets)
        min_dates = [dataset.min_date for dataset in datasets if dataset.min_date is not None]
        if len(min_dates) > 0:
            common_min_date = np.max(min_dates)
        else:
            common_min_date = None
        if _except_on_changing_date_bounds:
            assert common_min_date == self.min_date, "[ERROR] Encountered missing time stamps."
        else:
            if (common_min_date is not None) and (self.min_date is None or common_min_date > self.min_date):
                self.min_date = common_min_date
        max_dates = [dataset.max_date for dataset in datasets if dataset.max_date is not None]
        if len(max_dates) > 0:
            common_max_date = np.min(max_dates)
        else:
            common_max_date = None
        if _except_on_changing_date_bounds:
            assert common_max_date == self.max_date, "[ERROR] Encountered missing time stamps."
        else:
            if (common_max_date is not None) and (self.max_date is None or common_max_date < self.max_date):
                self.max_date = common_max_date
        self.data_groups.update({group_key: {dataset.name: dataset for dataset in datasets}})
        self._update_date_bounds()
        return self

    def _verify_data_group_inputs(self, group_key, datasets):
        assert isinstance(group_key, str), "[ERROR] Group keys must be of type string."
        assert group_key not in self.data_groups, "[ERROR] Group keys must be unique. Key <{}> is already existing.".format(
            group_key)
        if not isinstance(datasets, list):
            datasets = [datasets]
        for dataset in datasets:
            assert isinstance(dataset, (ConstantData, TimeVariateData))
            "[ERROR] Datasets must be given as TimeVariateData or ConstantData objects or a list thereof."
        data_names = [dataset.name for dataset in datasets]
        assert len(data_names) == len(
            np.unique(data_names)), "[ERROR] Dataset names must be unique within a common parameter group."

    def remove_data_group(self, group_key):
        if group_key in self.data_groups:
            del self.data_groups[group_key]
        return self

    def _update_date_bounds(self):
        for group in self.data_groups.values():
            for dataset in group.values():
                dataset.set_date_range(self.min_date, self.max_date)

    def set_date_range(self, min_date=None, max_date=None, datetime_format=None):
        min_date = _parse_date_input(min_date, datetime_format)
        max_date = _parse_date_input(max_date, datetime_format)
        _verify_date_bounds(min_date, max_date)
        self.min_date = min_date
        self.max_date = max_date
        self._update_date_bounds()
        return self

    def __len__(self):
        if self.min_date is None or self.max_date is None:
            return 1 if len(self.data_groups) > 0 else 0
        else:
            return int((self.max_date - self.min_date) / TEMPORAL_RESOLUTION) if len(self.data_groups) else 0

    def __getitem__(self, item):
        # dataset[item][1] is the integer indicating month index (from 1 for Januray to 12 for December).
        # dataset[item][2] name of the variable.
        # dataset[item][0] tensor data.
        return tuple(tuple(dataset[item] for dataset in group.values()) for group in self.data_groups.values())

    def get_data_names(self):
        return {
            group_key: tuple(dataset.name for dataset in group.values())
            for group_key, group in self.data_groups.items()
        }

    def get_named_item(self, item):
        return {
            group_key: {dataset.name: dataset[item] for dataset in group.values()}
            for group_key, group in self.data_groups.items()
        }

    def get_channel_count(self, group_key=None):
        if group_key is None:
            return {gk: self.get_channel_count(group_key=gk) for gk in self.data_groups}
        elif group_key in self.data_groups:
            return np.sum([dataset.get_channel_count() for dataset in self.data_groups[group_key].values()])
        else:
            raise Exception("[ERROR] Dataset does not contain a data group named <{}>.".format(group_key))

    def get_valid_time_stamps(self):
        return np.arange(self.min_date, self.max_date, TEMPORAL_RESOLUTION)
