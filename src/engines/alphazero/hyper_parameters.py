import os
import json
import random
import typing as t
import numpy as np
import torch
from dataclasses import dataclass, asdict
from src.games.commons import Game
from . import constants


@dataclass
class HyperParameters:
    _seed: t.Optional[int] = None
    num_res_blocks: int = 10
    num_hidden: int = 256
    mcts_num_iterations: int = 800
    mcts_puct_constant: float = 1.41
    dirichlet_noise_alpha: float = 0.3
    dirichlet_noise_epsilon: float = 0.25
    num_learning_iterations: int = 1_000
    num_self_play_games: int = 1_000
    num_epochs: int = 10
    batch_size: int = 256
    temperature: float = 1.0
    max_game_moves: int = 100

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
            constants.PROGRESS_FOLDER,
            game.__class__.__name__,
            "hyper-parameters.json",
        )

    def save_to_file(self, game: Game) -> None:
        file_path = self.make_file_path(game)
        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w") as file:
            json.dump(asdict(self), file, indent=2)

    @classmethod
    def load_from_file(cls, game: Game) -> tuple["HyperParameters", bool]:
        file_path = cls.make_file_path(game)

        if not os.path.exists(file_path):
            return HyperParameters(), True

        with open(file_path, "r") as file:
            data = json.load(file)

        return HyperParameters(**data), False
