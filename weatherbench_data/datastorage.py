"""Defines WeatherBenchNPYStorage class to read data in npy format.

# TODO: Test the script.
"""
import json
import os
from datetime import datetime
from itertools import chain

import numpy as np
import torch

from .fileconverter import DATETIME_FORMAT, TEMPORAL_RESOLUTION, \
    DIRECTORY_NAME_META_DATA, DIRECTORY_NAME_SAMPLE_DATA, FILE_NAME_META_DATA


class WeatherBenchNPYStorage(object):

    def __init__(self, path, domain_dimension=2):
        self._verify_path(path)
        self.path = os.path.abspath(path)
        self.domain_dimension=domain_dimension
        self.meta_data = None
        self._load_meta_data()
        assert len(self.meta_data["dims"]) >= domain_dimension
        self.name = self.meta_data["name"]
        self._is_time_variate = self.meta_data["time_variate"]
        self._samples = None
        self._read_sample_directory()

    @staticmethod
    def _verify_path(path):
        assert os.path.isdir(path), "[ERROR] <{}> is not a valid directory path.".format(path)
        contents = os.listdir(path)
        assert len(contents) == 2 and os.path.isdir(os.path.join(path, DIRECTORY_NAME_META_DATA)) and os.path.isdir(os.path.join(path, DIRECTORY_NAME_SAMPLE_DATA)),\
            "[ERROR] <{}> does not follow the expected folder structure of a WeatherBench parameter directory.".format(path)

    def _load_meta_data(self):
        # load meta data file
        with open(os.path.join(self.path, DIRECTORY_NAME_META_DATA, FILE_NAME_META_DATA + ".json"), "r") as f:
            self.meta_data = json.load(f)
        coordinates = self.meta_data["coords"]
        # convert coordinate value lists to numpy arrays
        for c in coordinates:
            c.update({"values": np.array(c["values"])})

    def _read_sample_directory(self):
        sample_directory = os.path.join(self.path, DIRECTORY_NAME_SAMPLE_DATA)
        if self._is_time_variate:
            sample_time_stamps = self._build_sample_index(sample_directory)
            self._verify_data_completeness(sample_time_stamps)
        else:
            self._load_constant_data(sample_directory)

    def _build_sample_index(self, sample_directory):
        sub_directories = [
            d for d in sorted(os.listdir(sample_directory))
            if os.path.isdir(os.path.join(sample_directory, d))
        ]
        samples = []
        time_stamps = []
        for sub_directory in sub_directories:
            contents_s = []
            contents_t = []
            for f in sorted(os.listdir(os.path.join(sample_directory, sub_directory))):
                if self._matches_sample_file_convention(f):
                    contents_s.append(os.path.join(sample_directory, sub_directory, f))
                    contents_t.append(self._file_name_to_datetime(f))
            samples.append(contents_s)
            time_stamps.append(contents_t)
        samples = np.array(list(chain.from_iterable(samples)))
        time_stamps = np.array(list(chain.from_iterable(time_stamps)))
        sorting_index = np.argsort(time_stamps)
        self._samples = (time_stamps[0], samples[sorting_index])
        return time_stamps[sorting_index]

    @staticmethod
    def _verify_data_completeness(sample_time_stamps):
        # verify that the data covers a comprehensive range of time stamps
        min_date = sample_time_stamps[0]
        max_date = sample_time_stamps[-1]
        assert len(sample_time_stamps) == int((max_date - min_date) / TEMPORAL_RESOLUTION) + 1, \
            "[ERROR] encountered missing data values."
        assert np.all(np.diff(sample_time_stamps) == TEMPORAL_RESOLUTION)

    def _matches_sample_file_convention(self, f):
        if not f.endswith(".npy"):
            return False
        f_split = f.split(".")
        if len(f_split) > 2:
            return False
        try:
            date = self._file_name_to_datetime(f)
        except:
            return False
        return True

    @staticmethod
    def _file_name_to_datetime(f):
        return np.datetime64(datetime.strptime(f.split(".")[0], DATETIME_FORMAT))

    def _load_constant_data(self, sample_directory):
        data = torch.tensor(np.load(os.path.join(sample_directory, "constant.npy")))
        self._samples = self._to_pytorch_standard_shape(data)

    def _to_pytorch_standard_shape(self, data):
        dim = len(data.shape)
        domain_dim = self.domain_dimension
        # care for channel dimensions
        if dim == domain_dim:
            data = data.unsqueeze(dim=0)
        elif dim > domain_dim + 1:
            data = torch.flatten(data, start_dim=0, end_dim=-(domain_dim + 1))
        # add batch (time) dimension
        return data.unsqueeze(dim=0)

    def __len__(self):
        if self._is_time_variate:
            return len(self._samples[1])
        else:
            return 1

    def __getitem__(self, item):
        if self._is_time_variate:
            idx = int((item - self._samples[0]) / TEMPORAL_RESOLUTION)
            data = torch.tensor(np.load(self._samples[1][idx]))
            return self._to_pytorch_standard_shape(data)
        else:
            return self._samples

    def get_valid_time_stamps(self):
        if self._is_time_variate:
            min_date = self._samples[0]
            return np.arange(min_date, min_date + len(self._samples[1]) * TEMPORAL_RESOLUTION, TEMPORAL_RESOLUTION)
        else:
            return None

    def is_time_variate(self):
        return self._is_time_variate

    def get_channel_count(self):
        count = 1
        for axis_length in self.meta_data["shape"][0:-self.domain_dimension]:
            count = count * axis_length
        return int(count)
