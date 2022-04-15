"""Defines NetCDF to NumPy convertor for WeatherBench data.
"""
import json
import os
import shutil
from datetime import datetime

import numpy as np
import xarray as xr

DATETIME_FORMAT = "%Y-%m-%d-%H"
TEMPORAL_RESOLUTION = np.timedelta64(1, "h")
DIRECTORY_NAME_META_DATA = "meta"
FILE_NAME_META_DATA = "metadata"
FILE_NAME_CONSTANT_DATA = "constant"
DIRECTORY_NAME_SAMPLE_DATA = "samples"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


class NetCDFNumpyConverter(object):

    def __init__(self, netcdf_extension=".nc", numpy_extension=".npy", datetime_format=DATETIME_FORMAT):
        self.NETCDF_EXTENSION = netcdf_extension
        self.NUMPY_EXTENSION = numpy_extension
        self.DATETIME_FORMAT = datetime_format
        self.source_directory = None
        self.data = None

    def set_source_directory(self, source_directory):
        self.source_directory = os.path.abspath(source_directory)
        return self

    def read_source_directory(self, source_directory=None, chunks=None, parallel=True):
        if self.source_directory is None and source_directory is None:
            raise Exception("[ERROR] Source directory must be set or given before it can be read.")
        elif source_directory is not None:
            self.set_source_directory(source_directory)
        data = xr.open_mfdataset(
            os.path.join(self.source_directory, "*" + self.NETCDF_EXTENSION), parallel=parallel, chunks=chunks
        )
        if chunks is None and "time" in data.dims:
            data = xr.open_mfdataset(
                os.path.join(self.source_directory, "*" + self.NETCDF_EXTENSION), parallel=parallel,
                chunks={"time": 12}
            )
        self.data = data
        return self

    def convert_to_pytorch_samples(self, target_directory, enable_batch_processing=False, batch_size=None, rename_vars=None, overwrite_previous=False):
        if self.source_directory is None:
            raise Exception("[ERROR] Source directory must be set and read pefore running the conversion.")
        if self.data is None:
            raise Exception("[ERROR] Source directory must be read pefore running the conversion.")
        print("[INFO] Converting NetCDF dataset at <{}> to PyTorch sample files.".format(self.source_directory))
        if enable_batch_processing:
            assert batch_size is not None, "[ERROR] If batch-processing is enabled, a batch size must be given"
        else:
            batch_size = 0
        if rename_vars is None:
           rename_vars = {} # use rename_vars for renaming the variable folders upon conversion
        else:
            assert isinstance(rename_vars, dict)
        data_vars = self.data.data_vars
        if len(data_vars) == 0:
            print("[INFO] Selected data set did not contain any variables. No further actions required.")
            return
        target_directory = os.path.abspath(target_directory)
        if not os.path.isdir(target_directory):
            os.makedirs(target_directory)
            print("[INFO] Created target directory at <{}>.".format(target_directory))
        for var_key in data_vars:
            print("[INFO] Processing data variable <{}>.".format(var_key))
            data_var = data_vars[var_key]
            meta_folder, samples_folder = self._create_new_var_directory(target_directory, var_key, rename_vars, overwrite_previous)
            self._convert_meta_data(data_var, meta_folder)
            self._convert_sample_data(data_var, samples_folder, batch_size)

    def _create_new_var_directory(self, target_directory, var_key, rename_vars, overwrite_previous):
        directory_name = var_key if var_key not in rename_vars else rename_vars[var_key]
        var_directory = os.path.join(target_directory, directory_name)
        if os.path.isdir(var_directory):
            if len(os.listdir(var_directory)) > 0 and not overwrite_previous:
                raise Exception("[ERROR] Tried to create variable directory at <{}> but directory existed and was found to be not empty.")
            else:
                print("[INFO] Removing previously existing variable directory.")
                shutil.rmtree(var_directory, ignore_errors=True)
        os.makedirs(var_directory)
        print("[INFO] Created new variable directory at <{}>.".format(var_directory))
        sub_directories = []
        for folder_name in ["meta", "samples"]:
            sub_dir = os.path.join(var_directory, folder_name)
            os.makedirs(sub_dir)
            sub_directories.append(sub_dir)
        return tuple(sub_directories)

    def _convert_meta_data(self, data_var, meta_folder):
        print("[INFO] Reading meta data.")
        meta_data = {}
        meta_data.update({"name": data_var.name})
        meta_data.update({"time_variate": "time" in list(data_var.dims)})
        meta_data.update({"dims": [dim_name for dim_name in data_var.dims if dim_name != "time"]})
        meta_data.update({"shape": [dim_length for dim_name, dim_length in zip(data_var.dims, data_var.data.shape) if dim_name != "time"]})
        meta_data.update({"coords": []})
        data_coords = self.data.coords
        for coord_key in data_coords:
            if coord_key != "time":
                axis = data_coords[coord_key]
                meta_data["coords"].append({
                    "name": axis.name,
                    "values": axis.values.tolist(),
                    "dims": list(axis.dims)
                })
        meta_data.update({"attrs": {**self.data.attrs, **data_var.attrs}})
        meta_data_file = os.path.join(meta_folder, FILE_NAME_META_DATA + ".json")
        with open(meta_data_file, "w") as f:
            json.dump(meta_data, f)
        print("[INFO] Stored meta data in <{}>.".format(meta_data_file))

    def _convert_sample_data(self, data_var, samples_folder, batch_size):
        if "time" in data_var.dims:
            self._convert_temporal_samples(data_var, samples_folder, batch_size)
        else:
            self._convert_constant(data_var, samples_folder)

    def _convert_temporal_samples(self, data_var, samples_folder, batch_size):
        print("[INFO] Converting temporal samples.")
        time_stamps = data_var["time"].values
        time_axis = tuple(data_var.dims).index("time")
        assert len(time_stamps) == len(np.unique(time_stamps)), "[ERROR] Encountered data variable with non-unique time stamps."
        batches = np.array_split(time_stamps, np.ceil(len(time_stamps) / batch_size))
        current_year = None
        storage_folder = None
        for sample_batch in batches:
            batch_data = np.array_split(data_var.sel(time=sample_batch).values, len(sample_batch), axis=time_axis)
            for time_stamp, data in zip(sample_batch, batch_data):
                time_stamp = self._numpy_date_to_datetime(time_stamp)
                if time_stamp.year != current_year:
                    current_year = time_stamp.year
                    storage_folder = os.path.join(samples_folder, "{}".format(current_year))
                    if not os.path.isdir(storage_folder):
                        os.makedirs(storage_folder)
                np.save(
                    os.path.join(storage_folder, self._file_name_from_time_stamp(time_stamp)),
                    np.squeeze(data, axis=time_axis)
                )

    def _numpy_date_to_datetime(self, time_stamp):
        total_seconds = (time_stamp - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
        return datetime.utcfromtimestamp(total_seconds)

    def _file_name_from_time_stamp(self, time_stamp):
        return time_stamp.strftime(self.DATETIME_FORMAT) + self.NUMPY_EXTENSION

    def _convert_constant(self, data_var, samples_folder):
        data = data_var.values
        np.save(
            os.path.join(samples_folder, FILE_NAME_CONSTANT_DATA + self.NUMPY_EXTENSION),
            data
        )
