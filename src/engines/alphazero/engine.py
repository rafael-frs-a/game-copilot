import typing as t
import random
import torch
import numpy as np
from src import utils
from src.games.commons import Game
from src.engines import commons
from src.engines.alphazero.game import AlphaZeroGame


class HyperParameters:
    def __init__(self) -> None:
        self._seed: t.Optional[int] = None

    @property
    def seed(self) -> t.Optional[int]:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        if self._seed is not None:
            raise Exception("Cannot change seed")

        self._seed = value
        self.apply_seed()

    def apply_seed(self) -> None:
        if self._seed is None:
            return

        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AlphaZero(commons.Engine):
    def __init__(self, game: Game) -> None:
        if not isinstance(game, AlphaZeroGame):
            raise ValueError("Game not supported by AlphaZero")

        super().__init__(game)

    def set_seed(self) -> None:
        # Ask for numeric seed
        # This allows deterministic reproducibility
        # Inform nothing if we want to skip it
        while True:
            seed = utils.prompt_input("Enter a numeric seed (optional): ")

            if seed == "":
                break

            try:
                self.hyper_parameters.seed = int(seed)
                break
            except ValueError:
                print(
                    "Invalid input. Please enter a valid number or press enter to skip this step"
                )

    def setup(self) -> None:
        self.hyper_parameters = HyperParameters()
        self.set_seed()

    def suggest_move(self) -> None:
        pass
