import abc
import numpy as np
from functools import cache
from numpy import typing as npt
from src.games import commons as games_commons


class AlphaZeroGame(games_commons.Game):
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
    def make_state_input_tensor(
        self, state: games_commons.GameState
    ) -> npt.NDArray[np.float32]:
        pass

    def get_state_value(
        self, state: games_commons.GameState, player: games_commons.Player, value: float
    ) -> float:
        # This works for two players zero sum games, like tic-tac-toe and chess
        # Other games might need a different approach
        if player == state.current_player:
            return value

        return -value
