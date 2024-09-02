import typing as t
import torch
from numpy import typing as npt
from . import constants


def make_torch_tensor(array: npt.NDArray[t.Any]) -> torch.Tensor:
    return torch.Tensor(array).to(constants.DEVICE)
