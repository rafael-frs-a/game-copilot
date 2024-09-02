import os
import json
import random
import typing as t
import numpy as np
import torch
from dataclasses import dataclass, asdict
from src import utils
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
    temperature: float = 1.0  # Temperature of 1 has no effect in move selection
    max_game_moves: int = 100
    save_input_history: bool = True

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

    def prompt_input(self, message: str) -> str:
        return utils.prompt_input(message, self.save_input_history)

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

    def set_seed(self) -> None:
        # Ask for numeric seed
        # This allows deterministic reproducibility
        # Inform nothing if we want to skip it
        while True:
            input_value = self.prompt_input("Enter a numeric seed (optional): ")

            if input_value == "":
                break

            try:
                self.seed = int(input_value)
                break
            except ValueError:
                print(
                    "Invalid input. Please enter a valid number or press enter to skip this step"
                )

    def set_num_hidden_layers(self) -> None:
        while True:
            input_value = self.prompt_input(
                f"Enter the number of hidden layers (optional, default {self.num_hidden}): "
            )

            if not input_value:
                break

            try:
                self.num_hidden = int(input_value)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")

    def set_num_res_blocks(self) -> None:
        while True:
            input_value = self.prompt_input(
                f"Enter the number of residual blocks (optional, default {self.num_res_blocks}): "
            )

            if not input_value:
                break

            try:
                self.num_res_blocks = int(input_value)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")

    def set_mcts_iterations(self) -> None:
        while True:
            input_value = self.prompt_input(
                f"Enter the number of MCTS iterations (optional, default {self.mcts_num_iterations}): "
            )

            if not input_value:
                break

            try:
                self.mcts_num_iterations = max(0, int(input_value))
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")

    def set_puct(self) -> None:
        while True:
            input_value = self.prompt_input(
                f"Enter the PUCT constant (optional, default {self.mcts_puct_constant}): "
            )

            if not input_value:
                break

            try:
                self.mcts_puct_constant = float(input_value)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")

    def set_dirichlet_alpha(self) -> None:
        while True:
            input_value = self.prompt_input(
                f"Enter the Dirichlet noise alpha (optional, default {self.dirichlet_noise_alpha}): "
            )

            if not input_value:
                break

            try:
                self.dirichlet_noise_alpha = float(input_value)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")

    def set_dirichlet_epsilon(self) -> None:
        while True:
            input_value = self.prompt_input(
                f"Enter the Dirichlet noise epsilon (optional, default {self.dirichlet_noise_epsilon}): "
            )

            if not input_value:
                break

            try:
                self.dirichlet_noise_epsilon = float(input_value)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")

    def set_num_learning_iterations(self) -> None:
        while True:
            input_value = self.prompt_input(
                f"Enter the number of learning iterations (optional, default {self.num_learning_iterations}): "
            )

            if not input_value:
                break

            try:
                self.num_learning_iterations = max(0, int(input_value))
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")

    def set_num_self_play_games(self) -> None:
        while True:
            input_value = self.prompt_input(
                f"Enter the number of self play games (optional, default {self.num_self_play_games}): "
            )

            if not input_value:
                break

            try:
                self.num_self_play_games = max(0, int(input_value))
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")

    def set_num_epochs(self) -> None:
        while True:
            input_value = self.prompt_input(
                f"Enter the number of epochs (optional, default {self.num_epochs}): "
            )

            if not input_value:
                break

            try:
                self.num_epochs = max(0, int(input_value))
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")

    def set_batch_size(self) -> None:
        while True:
            input_value = self.prompt_input(
                f"Enter the number of batch size (optional, default {self.batch_size}): "
            )

            if not input_value:
                break

            try:
                self.batch_size = max(0, int(input_value))
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")

    def set_temperature(self) -> None:
        while True:
            input_value = self.prompt_input(
                f"Enter the move selection temperature (optional, default {self.temperature}): "
            )

            if not input_value:
                break

            try:
                self.temperature = float(input_value)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")

    def set_max_game_moves(self) -> None:
        while True:
            input_value = self.prompt_input(
                f"Enter the number of max. moves per game (optional, default {self.max_game_moves}): "
            )

            if not input_value:
                break

            try:
                self.max_game_moves = max(0, int(input_value))
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")
