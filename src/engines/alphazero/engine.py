import typing as t
import numpy as np
from src.games.commons import Game
from src.engines import commons
from src.engines.alphazero.game import AlphaZeroGame
from src.engines.alphazero.model import GameNet
from src.engines.alphazero.hyper_parameters import HyperParameters
from src.engines.alphazero.mcts import MCTS


class AlphaZero(commons.Engine):
    def __init__(self, game: Game) -> None:
        if not isinstance(game, AlphaZeroGame):
            raise ValueError("Game not supported by AlphaZero")

        super().__init__(game)

    def load_hyperparameters(self) -> None:
        self.hyper_parameters, created = HyperParameters.load_from_file(self.game)

        if created:
            self.hyper_parameters.set_num_hidden_layers()
            self.hyper_parameters.set_num_res_blocks()

    def load_model(self) -> None:
        self.model = GameNet(
            t.cast(AlphaZeroGame, self.game),
            self.hyper_parameters.num_res_blocks,
            self.hyper_parameters.num_hidden,
        )

    def load_mcts(self) -> None:
        self.mcts = MCTS(
            t.cast(AlphaZeroGame, self.game), self.hyper_parameters, self.model
        )

    def setup(self) -> None:
        self.load_hyperparameters()
        self.hyper_parameters._seed = None
        self.hyper_parameters.set_seed()
        self.hyper_parameters.set_mcts_iterations()
        self.load_model()
        self.load_mcts()
        self.model.eval()
        self.all_game_moves = t.cast(
            AlphaZeroGame, self.game
        ).generate_all_possible_moves()

    def suggest_move(self) -> None:
        move_probs, value = self.mcts.search()
        winning_changes = max(value, 0)
        move_idx = np.argmax(move_probs)
        suggested_move = self.all_game_moves[move_idx]
        print(f"Suggested move: {suggested_move}. Winning chances: {winning_changes}")
