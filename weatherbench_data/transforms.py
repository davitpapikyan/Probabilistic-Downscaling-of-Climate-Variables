"""Defines various transformations for WeatherBench data.
"""
import numpy as np
import torch
import torch.nn as nn


class Transform(nn.Module):
    def __init__(self, requires_fit, exclude_at_evaluation=False):
        super(Transform, self).__init__()
        self.requires_fit = requires_fit
        self.exclude_at_evaluation = exclude_at_evaluation

    def transform(self, data):
        raise NotImplementedError()

    def out_channels(self, in_channels):
        return in_channels

    def forward(self, data):
        return self.transform(data)

    def is_data_adaptive(self):
        return self.requires_fit

    def summarize(self):
        return {"transform_type": self.__class__.__name__}


class ReversibleTransform(Transform):
    def __init__(self, requires_fit, exclude_at_evaluation=False):
        super(ReversibleTransform, self).__init__(
            requires_fit=requires_fit, exclude_at_evaluation=exclude_at_evaluation
        )

    def transform(self, data):
        raise NotImplementedError()

    def revert(self, data):
        raise NotImplementedError()


class AdaptiveReversibleTransform(ReversibleTransform):
    def __init__(self, exclude_at_evaluation=False):
        super(AdaptiveReversibleTransform, self).__init__(
            requires_fit=True, exclude_at_evaluation=exclude_at_evaluation
        )
        self._data_source = None

    def fit(self, dataset, batch_size=None, previous_transforms=None, disable_fitting_mode=False):
        if self._data_source is not None:
            raise Exception("[ERROR] Fit should only be called once on adaptive transform objects.")
        if previous_transforms is not None:
            assert isinstance(previous_transforms, list)
            for t in previous_transforms:
                assert isinstance(t, Transform)
        if not dataset.is_time_variate():
            self._fit_to_batch(dataset, [0], previous_transforms)
        else:
            in_fitting_mode = dataset.get_fitting_mode()
            if in_fitting_mode != disable_fitting_mode:
                dataset.set_fitting_mode(disable_fitting_mode)
            if batch_size is None:
                self._fit_to_batch(dataset, np.arange(len(dataset)), previous_transforms)
            else:
                assert isinstance(batch_size, int)
                idx = np.arange(len(dataset))
                batches = np.array_split(idx, np.ceil(len(idx) / batch_size))
                for idx_batch in batches:
                    self._fit_to_batch(dataset, idx_batch, previous_transforms)
            dataset.set_fitting_mode(in_fitting_mode)

        self._fill_data_source(dataset, previous_transforms)
        return self

    def _fit_to_batch(self, dataset, batch, previous_transforms):
        for data in dataset.get_batch(batch):
            if previous_transforms is not None:
                for t in previous_transforms:
                    data = t.transform(data)
            self._update_parameters(data)

    def _fill_data_source(self, dataset, previous_transforms):
        self._data_source = dataset.summarize()
        if previous_transforms is not None:
            self._data_source.update({
                "previous_transforms": [
                    t.summarize() for t in reversed(previous_transforms)
                ]
            })

    def clear_data_source(self):
        self._data_source = None

    def _update_parameters(self, data):
        raise NotImplementedError()

    def transform(self, data):
        raise NotImplementedError()

    def revert(self, data):
        raise NotImplementedError()

    def summarize(self):
        summary = super(AdaptiveReversibleTransform, self).summarize()
        summary.update({"data_source": self._data_source})
        return summary


class StandardScaling(AdaptiveReversibleTransform):
    def __init__(self, unbiased=True):
        super(StandardScaling, self).__init__(exclude_at_evaluation=False)
        self._count = 0
        self._bias_correction = int(unbiased)
        self.register_buffer("_mean", None)
        self.register_buffer("_squared_differences", None)

    def _std(self):
        return torch.sqrt(self._squared_differences / (self._count - self._bias_correction))

    def transform(self, data):
        return (data - self._mean) / self._std()

    def revert(self, data):
        return (self._std() * data) + self._mean

    def _update_parameters(self, data):
        data_stats = self._compute_stats(data)
        if self._mean is None:
            self._count, self._mean, self._squared_differences = data_stats
            return self
        return self._update_stats(*data_stats)

    def _compute_stats(self, data):
        raise NotImplementedError()

    def _update_stats(self, data_count, data_mean, data_squared_differences):
        new_count = self._count + data_count
        self._squared_differences += data_squared_differences
        self._squared_differences += (data_mean - self._mean)**2 * ((data_count * self._count) / new_count)
        self._mean = ((self._count * self._mean) + (data_count * data_mean)) / new_count
        self._count = new_count
        return self


class LocalStandardScaling(StandardScaling):

    def _compute_stats(self, data):
        data_count = data.shape[0]
        data_mean = torch.mean(data, dim=0, keepdim=True)
        return data_count, data_mean, torch.sum(torch.square(data - data_mean), dim=0, keepdim=True)


class LatitudeStandardScaling(StandardScaling):

    def _compute_stats(self, data):
        shape = data.shape
        data_count = shape[0] * shape[3]
        data_mean = torch.mean(data, dim=(0, 3), keepdim=True)
        return data_count, data_mean, torch.sum(torch.square(data - data_mean), dim=(0, 3), keepdim=True)


class GlobalStandardScaling(StandardScaling):

    def _compute_stats(self, data):
        shape = data.shape
        data_count = shape[0] * shape[2] * shape[3]
        data_mean = torch.mean(data, dim=(0, 2, 3), keepdim=True)
        return data_count, data_mean, torch.sum(torch.square(data - data_mean), dim=(0, 2, 3), keepdim=True)


class AngularTransform(ReversibleTransform):
    def __init__(self, mode="deg", clamp=True):
        super(AngularTransform, self).__init__(requires_fit=False, exclude_at_evaluation=False)
        assert mode in ["deg", "rad"], "[ERROR] Mode of angular transform must be \"deg\" or \"rad\"."
        self._deg = (mode == "deg")
        self._clamp = clamp

    def transform(self, data):
        output = data
        if self._deg:
            output = torch.deg2rad(output)
        return self._transform(output)

    @staticmethod
    def _transform(data):
        raise NotImplementedError()

    def revert(self, data):
        output = data
        if self._clamp:
            output = torch.clamp(output, min=-1, max=1)
        output = self._revert(output)
        if self._deg:
            output = torch.rad2deg(output)
        return output

    @staticmethod
    def _revert(data):
        raise NotImplementedError()


class Cosine(AngularTransform):

    @staticmethod
    def _transform(data):
        return torch.cos(data)

    @staticmethod
    def _revert(data):
        return torch.acos(data)


class Sine(AngularTransform):

    @staticmethod
    def _transform(data):
        return torch.sin(data)

    @staticmethod
    def _revert(data):
        return torch.asin(data)


class PolarCoordinates(AngularTransform):

    @staticmethod
    def _transform(data):
        return torch.cat([torch.cos(data), torch.sin(data)], dim=1)

    @staticmethod
    def _revert(data):
        data = torch.chunk(data, 2, dim=1)
        return torch.angle(data[0] + 1j * data[1])

    def out_channels(self, in_channels):
        return 2 * in_channels


class GlobalRandomOffset(Transform):
    def __init__(self, minimum=0, maximum=1):
        assert maximum > minimum
        super(GlobalRandomOffset, self).__init__(requires_fit=False, exclude_at_evaluation=True)
        self._min = minimum
        self._max = maximum

    def transform(self, data):
        offset = torch.rand(data.shape[0], 1, 1, 1, device=data.device, dtype=data.dtype)
        return data + ((self._max - self.min) * offset + self._min)
