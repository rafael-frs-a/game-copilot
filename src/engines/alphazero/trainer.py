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
        active_game_states = [
            self.game.current_state
        ] * self.hyper_parameters.num_self_play_games
        games_history: list[
            list[
                tuple[
                    games_commons.GameState,
                    games_commons.Player,
                    npt.NDArray[np.float32],
                ]
            ]
        ] = [[] for _ in range(len(active_game_states))]
        memory: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], float]] = (
            []
        )

        while active_game_states:
            moves_probs, _ = self.mcts.search(active_game_states)

            if self.hyper_parameters.temperature:
                moves_probs = moves_probs ** (1 / self.hyper_parameters.temperature)

            next_active_game_states: list[games_commons.GameState] = []
            next_games_history: list[
                list[
                    tuple[
                        games_commons.GameState,
                        games_commons.Player,
                        npt.NDArray[np.float32],
                    ]
                ]
            ] = []

            for idx, active_state in enumerate(active_game_states):
                player = active_state.get_next_player()
                games_history[idx].append((active_state, player, moves_probs[idx]))
                move = np.random.choice(self.all_game_moves, p=moves_probs[idx])
                result = self.game.evaluate_move(active_state, move)

                if not result.success or not result.data:
                    raise Exception(
                        "Invalid configuration. Engine selected an invalid move"
                    )

                new_state = result.data

                if (
                    not self.game.is_terminal(new_state)
                    and new_state.move_count < self.hyper_parameters.max_game_moves
                ):
                    next_active_game_states.append(new_state)
                    next_games_history.append(games_history[idx])
                    continue

                winner = new_state.winner

                for state, player, move_probs in games_history[idx]:
                    state_tensor = self.game.make_state_input_tensor(state)
                    value = 0.0

                    if winner:
                        value = 1.0 if player == winner else -1.0

                    memory.append((state_tensor, move_probs, value))

            active_game_states = next_active_game_states
            games_history = next_games_history

        return memory

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
            self.model.eval()
            memory = self.self_play()
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
