import abc
import typing as t
import numpy as np
from functools import cache
from numpy import typing as npt
from src.games.commons import GameState


if t.TYPE_CHECKING:
    from src.games.commons import Game


class AlphaZeroGame(Game):
    @property
    @abc.abstractmethod
    def input_tensor_dimensions(self) -> tuple[int, int, int]:
        # The dimensions are, in order:
        # 1. Channels of boards containing game specific elements, like pieces, players, and auxiliary conditions
        # 2. Board height
        # 3. Board width
        pass

    @cache
    @abc.abstractmethod
    def generate_all_possible_moves(self) -> npt.NDArray[np.str_]:
        pass

    @cache
    @abc.abstractmethod
    def make_state_input_tensor(self, state: GameState) -> npt.NDArray[np.float32]:
        pass
