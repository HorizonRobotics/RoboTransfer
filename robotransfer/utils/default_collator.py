import numpy as np
import torch

# from .. import utils

def stack_data(data_list, pad_value=0, is_equal=False):
    equal_shape = True
    for data in data_list:
        if data.shape != data_list[0].shape:
            equal_shape = False
            break
    if is_equal:
        assert equal_shape
    if equal_shape:
        if isinstance(data_list[0], np.ndarray):
            new_data = np.stack(data_list)
        elif isinstance(data_list[0], torch.Tensor):
            new_data = torch.stack(data_list)
        else:
            assert False
    else:
        ndim = data_list[0].ndim
        dtype = data_list[0].dtype
        new_data_shape = [len(data_list)]
        for dim in range(ndim):
            new_data_shape.append(max(data.shape[dim] for data in data_list))
        if isinstance(data_list[0], np.ndarray):
            new_data = np.full(tuple(new_data_shape), pad_value, dtype=dtype)
        elif isinstance(data_list[0], torch.Tensor):
            new_data = torch.full(tuple(new_data_shape), pad_value, dtype=dtype)
        else:
            assert False
        for i, data in enumerate(data_list):
            if ndim == 1:
                new_data[i][: data.shape[0]] = data
            elif ndim == 2:
                new_data[i][: data.shape[0], : data.shape[1]] = data
            elif ndim == 3:
                new_data[i][: data.shape[0], : data.shape[1], : data.shape[2]] = data
            elif ndim == 4:
                new_data[i][: data.shape[0], : data.shape[1], : data.shape[2], : data.shape[3]] = data
            else:
                assert False
    return new_data

class DefaultCollator:
    def __init__(self, is_equal=False):
        self.is_equal = is_equal

    def __call__(self, batch):
        batch_dict = dict()
        if isinstance(batch, list):
            for key in batch[0]:
                batch_dict[key] = self._collate([d[key] for d in batch])
        elif isinstance(batch, dict):
            for key in batch:
                batch_dict[key] = self._collate(batch[key])
        else:
            assert False
        return batch_dict

    def _collate(self, batch):
        if isinstance(batch, (list, tuple)):
            if isinstance(batch[0], torch.Tensor):
                batch = stack_data(batch, is_equal=self.is_equal)
            elif isinstance(batch[0], np.ndarray):
                batch = stack_data(batch, is_equal=self.is_equal)
                batch = torch.from_numpy(batch)
            elif isinstance(batch[0], (np.bool_, np.number, np.object_)):
                batch = torch.as_tensor(batch)
            elif isinstance(batch[0], dict):
                batch = {key: self._collate([d[key] for d in batch]) for key in batch[0]}
            elif isinstance(batch[0], (list, tuple)):
                batch = type(batch[0])([self._collate(d) for d in zip(*batch)])
        elif isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        return batch
