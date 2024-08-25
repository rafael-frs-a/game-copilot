import torch
import typing as t
from functools import cache


@cache
def get_device() -> t.Literal["cpu", "cuda"]:
    return "cuda" if torch.cuda.is_available() else "cpu"
