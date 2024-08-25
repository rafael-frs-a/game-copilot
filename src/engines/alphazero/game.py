import abc
from functools import cache


class AlphaZeroGame(abc.ABC):
    @property
    @abc.abstractmethod
    def input_tensor_dimensions(self) -> tuple[int, int, int]:
        # The dimensions are, in order:
        # 1. Board height
        # 2. Board width
        # 3. Board planes containing game specific elements, like pieces, players, and auxiliary conditions
        pass

    @cache
    @abc.abstractmethod
    def generate_all_possible_moves(self) -> list[str]:
        pass
