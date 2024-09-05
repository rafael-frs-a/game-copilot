import typing as t
import torch
from numpy import typing as npt
from . import constants


def make_torch_tensor(array: npt.NDArray[t.Any], use_cuda: bool = True) -> torch.Tensor:
    tensor = torch.Tensor(array)

    if use_cuda:
        tensor = tensor.to(constants.DEVICE)

    return tensor
