import typing as t
from src import utils
from src.games.commons import Game
from src.engines import commons
from src.engines.alphazero.game import AlphaZeroGame
from src.engines.alphazero.model import GameNet
from src.engines.alphazero.hyper_parameters import HyperParameters


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

    def set_mcts_iterations(self) -> None:
        while True:
            iterations = utils.prompt_input("Enter the number of MCTS iterations: ")

            try:
                self.hyper_parameters.mcts_num_iterations = max(0, int(iterations))
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")

    def load_hyperparameters(self) -> None:
        self.hyper_parameters, _ = HyperParameters.load_from_file(self.game)

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
        self.set_mcts_iterations()
        self.load_model()
        self.model.eval()
        self.all_game_moves = self.game.generate_possible_moves()

    def suggest_move(self) -> None:
        pass
