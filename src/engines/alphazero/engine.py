import os
import typing as t
import json
import random
import torch
import numpy as np
from dataclasses import dataclass, asdict
from src import utils
from src.games.commons import Game
from src.engines import commons
from src.engines.alphazero.game import AlphaZeroGame
from src.engines.alphazero.model import GameNet


@dataclass
class HyperParameters:
    _seed: t.Optional[int] = None
    num_res_blocks: int = 10
    num_hidden: int = 256

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

    @staticmethod
    def get_current_dir() -> str:
        return os.path.dirname(os.path.realpath(__file__))

    @staticmethod
    def make_file_path(game: Game) -> str:
        return os.path.join(
            HyperParameters.get_current_dir(),
            "trained-weights",
            game.__class__.__name__,
            "hyper_parameters.json",
        )

    def save_to_file(self, game: Game) -> None:
        file_path = self.make_file_path(game)
        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w") as file:
            json.dump(asdict(self), file, indent=2)

    @classmethod
    def load_from_file(cls, game: Game) -> "HyperParameters":
        file_path = cls.make_file_path(game)

        if not os.path.exists(file_path):
            return HyperParameters()

        with open(file_path, "r") as file:
            data = json.load(file)

        return HyperParameters(**data)


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

    def load_hyperparameters(self) -> None:
        self.hyper_parameters = HyperParameters.load_from_file(self.game)

    def load_model(self) -> None:
        self.model = GameNet(
            t.cast(AlphaZeroGame, self.game),
            self.hyper_parameters.num_res_blocks,
            self.hyper_parameters.num_hidden,
        )

    def setup(self) -> None:
        self.load_hyperparameters()
        self.hyper_parameters.seed = None
        self.set_seed()
        self.load_model()

    def suggest_move(self) -> None:
        pass
