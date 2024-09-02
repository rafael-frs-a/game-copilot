import random
import numpy as np
from enum import Enum
from tqdm import trange
from numpy import typing as npt
from torch.nn import functional as F
from src import utils
from src.games import commons as games_commons
from src.games.tic_tac_toe import TicTacToe
from src.games.chess import Chess
from src.engines.alphazero import utils as alphazero_utils
from src.engines.alphazero.game import AlphaZeroGame
from src.engines.alphazero.hyper_parameters import HyperParameters
from src.engines.alphazero.optimizer import Optimizer
from src.engines.alphazero.mcts import MCTS
from src.engines.alphazero.model import GameNet


class AlphaZeroGameType(Enum):
    TIC_TAC_TOE = "tic-tac-toe"
    CHESS = "chess"


def make_game(game_type: str) -> AlphaZeroGame:
    if game_type == AlphaZeroGameType.TIC_TAC_TOE.value:
        return TicTacToe()

    if game_type == AlphaZeroGameType.CHESS.value:
        return Chess()

    raise ValueError("The selected game is not supported")


class AlphaZeroTrainer:
    def select_game(self) -> None:
        game_options = [game_type.value for game_type in AlphaZeroGameType]

        if len(game_options) == 1:
            print(f'Selecting "{game_options[0]}"')
            self.game = make_game(game_options[0])
            return

        while True:
            try:
                game = utils.prompt_input(
                    f"Enter the game ({', '.join(game_options)}): ", False
                )
                self.game = make_game(game)
                break
            except ValueError as exc:
                print(str(exc))

    def load_hyperparameters(self) -> None:
        self.hyper_parameters, created = HyperParameters.load_from_file(self.game)
        self.hyper_parameters.save_input_history = False

        if created:
            self.hyper_parameters.set_seed()
            self.hyper_parameters.set_num_hidden_layers()
            self.hyper_parameters.set_num_res_blocks()
            self.hyper_parameters.set_mcts_iterations()
            self.hyper_parameters.set_puct()
            self.hyper_parameters.set_dirichlet_alpha()
            self.hyper_parameters.set_dirichlet_epsilon()
            self.hyper_parameters.set_num_learning_iterations()
            self.hyper_parameters.set_num_self_play_games()
            self.hyper_parameters.set_num_epochs()
            self.hyper_parameters.set_batch_size()
            self.hyper_parameters.set_temperature()
            self.hyper_parameters.set_max_game_moves()

    def load_mcts(self) -> None:
        self.mcts = MCTS(self.game, self.hyper_parameters, self.model)

    def load_model(self) -> None:
        self.model = GameNet(
            self.game,
            self.hyper_parameters.num_res_blocks,
            self.hyper_parameters.num_hidden,
        )

    def load_optimizer(self) -> None:
        self.optimizer = Optimizer(self.model)

    def self_play(
        self,
    ) -> list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], float]]:
        self.game.setup()
        game_history: list[
            tuple[
                games_commons.GameState, games_commons.Player, npt.NDArray[np.float32]
            ]
        ] = []
        result: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], float]] = (
            []
        )

        while (
            not self.game.is_terminal(self.game.current_state)
            and self.game.current_state.move_count
            < self.hyper_parameters.max_game_moves
        ):
            state = self.game.current_state
            player = state.get_next_player()
            move_probs, _ = self.mcts.search()
            game_history.append((state, player, move_probs))

            if self.hyper_parameters.temperature:
                move_probs = move_probs ** (1 / self.hyper_parameters.temperature)

            move = np.random.choice(self.all_game_moves, p=move_probs)
            evaluation_result = self.game.evaluate_move(state, move)

            if not evaluation_result.success or not evaluation_result.data:
                raise Exception(
                    "Invalid configuration. Engine selected an invalid move"
                )

            self.game.apply_state(evaluation_result.data)

        winner = self.game.current_state.winner

        for state, player, move_probs in game_history:
            state_tensor = self.game.make_state_input_tensor(state)
            value = 0.0

            if winner:
                value = 1.0 if player == winner else -1.0

            result.append((state_tensor, move_probs, value))

        return result

    def train(
        self,
        memory: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], float]],
    ) -> None:
        random.shuffle(memory)

        for batch_idx in range(0, len(memory), self.hyper_parameters.batch_size):
            sample = memory[
                batch_idx : min(  # noqa
                    batch_idx + self.hyper_parameters.batch_size, len(memory) - 1
                )
            ]
            state, policy_targets, value_targets = zip(*sample)
            state_array, policy_targets_array, value_targets_array = (
                np.array(state),
                np.array(policy_targets),
                np.array(value_targets).reshape(-1, 1),
            )
            state_tensor = alphazero_utils.make_torch_tensor(state_array)
            policy_targets_tensor = alphazero_utils.make_torch_tensor(
                policy_targets_array
            )
            value_targets_tensor = alphazero_utils.make_torch_tensor(
                value_targets_array
            )

            out_policy, out_value = self.model(state_tensor)

            policy_loss = F.cross_entropy(out_policy, policy_targets_tensor)
            value_loss = F.mse_loss(out_value, value_targets_tensor)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            self.optimizer.step()

    def learn(self) -> None:
        for _ in trange(
            self.hyper_parameters.num_learning_iterations, desc="Learning iterations"
        ):
            memory: list[
                tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], float]
            ] = []
            self.model.eval()

            for _ in trange(
                self.hyper_parameters.num_self_play_games, desc="Self-play games"
            ):
                memory += self.self_play()

            self.model.train()

            for _ in trange(self.hyper_parameters.num_epochs, desc="Training weights"):
                self.train(memory)

    def save_progress(self) -> None:
        self.hyper_parameters.save_to_file(self.game)
        self.model.save_state()
        self.optimizer.save_state()

    def run(self) -> None:
        self.select_game()
        self.all_game_moves = self.game.generate_all_possible_moves()
        self.load_hyperparameters()
        self.load_model()
        self.load_optimizer()
        self.load_mcts()

        try:
            self.learn()
            self.save_progress()
        except KeyboardInterrupt:
            self.save_progress()
