from functools import cache
from mypy_extensions import trait


@trait
class AlphaZeroGame:
    @property
    def input_tensor_dimensions(self) -> tuple[int, int, int]:
        # The dimensions are, in order:
        # 1. Board height
        # 2. Board width
        # 3. Board planes containing game specific elements, like pieces, players, and auxiliary conditions
        raise NotImplementedError

    @cache
    def generate_all_possible_moves(self) -> list[str]:
        raise NotImplementedError
