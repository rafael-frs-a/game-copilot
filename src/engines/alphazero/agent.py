import os
import pickle
import glob
import random
import psutil
import numpy as np
from enum import Enum
from datetime import datetime
from functools import cached_property
from tqdm import trange
from numpy import typing as npt
from torch import multiprocessing as mp
from torch.nn import functional as F
from src import utils
from src.games.tic_tac_toe import TicTacToe
from src.games.chess import Chess
from src.engines.alphazero import constants
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


class AlphaZeroAgent:
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
            self.hyper_parameters.set_num_cpus()
        elif self.hyper_parameters._seed is not None:
            seed, self.hyper_parameters._seed = (
                self.hyper_parameters._seed,
                None,
            )  # Force apply seed
            self.hyper_parameters.seed = seed

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

    @cached_property
    def training_folder(self) -> str:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        folder = os.path.join(
            current_dir,
            constants.PROGRESS_FOLDER,
            self.model.game.__class__.__name__,
            constants.TRAINING_FOLDER,
        )
        os.makedirs(folder, exist_ok=True)
        return folder

    def self_play(self, worker_idx: int, num_self_play_games: int) -> None:
        self.game.setup()
        active_games = {
            game_idx: self.game.current_state for game_idx in range(num_self_play_games)
        }
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

        while active_games:
            current_active_games = list(active_games.items())
            active_states = [
                current_active_game[1] for current_active_game in current_active_games
            ]
            moves_probs, _ = self.mcts.search(active_states)

            if self.hyper_parameters.temperature:
                moves_probs = moves_probs ** (1 / self.hyper_parameters.temperature)

            for idx, (game_idx, active_state) in enumerate(current_active_games):
                player = active_state.get_next_player()
                move_probs = moves_probs[idx]
                game_history_file_path = os.path.join(
                    self.training_folder,
                    f"game-history-{worker_idx}-{game_idx}-{timestamp}.pkl",
                )

                with open(game_history_file_path, "ab") as f:
                    state_array = self.game.make_state_input_tensor(active_state)
                    history = (state_array, player.__hash__(), move_probs)
                    pickle.dump(history, f)

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
                    active_games[game_idx] = new_state
                    continue

                active_games.pop(game_idx)
                winner = new_state.winner
                game_history: list[
                    tuple[
                        npt.NDArray[np.float32],
                        int,
                        npt.NDArray[np.float32],
                    ]
                ] = []

                with open(game_history_file_path, "rb") as f:
                    while True:
                        try:
                            game_history.append(pickle.load(f))
                        except EOFError:
                            break

                game_memory_file_path = os.path.join(
                    self.training_folder,
                    f"memory-{worker_idx}-{game_idx}-{timestamp}.pkl",
                )
                memory: list[
                    tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], float]
                ] = []

                for state_tensor, player_hash, move_probs in game_history:
                    value = 0.0

                    if winner:
                        value = 1.0 if player_hash == winner.__hash__() else -1.0

                    memory.append((state_tensor, move_probs, value))

                with open(game_memory_file_path, "wb") as f:
                    pickle.dump(memory, f)

    def train(
        self,
    ) -> None:
        file_list = glob.glob(os.path.join(self.training_folder, "memory-*.pkl"))
        memory: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], float]] = (
            []
        )

        for file in file_list:
            with open(file, "rb") as f:
                memory.extend(pickle.load(f))

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

    def clear_training_folder(self) -> None:
        for filename in os.listdir(self.training_folder):
            if filename.endswith(".pkl"):
                file_path = os.path.join(self.training_folder, filename)
                os.unlink(file_path)

    def learn(self) -> None:
        for _ in trange(
            self.hyper_parameters.num_learning_iterations, desc="Learning iterations"
        ):
            self.clear_training_folder()
            self.model.eval()
            processes: list[mp.Process] = []
            cpu_count = min(
                self.hyper_parameters.num_cpus, psutil.cpu_count(logical=False)
            )
            base_batch_size = self.hyper_parameters.num_self_play_games // cpu_count
            remainder = self.hyper_parameters.num_self_play_games % cpu_count

            for worker_idx in range(cpu_count):
                num_self_play_games = (
                    base_batch_size + 1 if worker_idx < remainder else base_batch_size
                )

                if num_self_play_games <= 0:
                    break

                p = mp.Process(
                    target=self.self_play, args=(worker_idx, num_self_play_games)
                )
                p.daemon = True
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            self.model.train()

            for _ in trange(self.hyper_parameters.num_epochs, desc="Training epochs"):
                self.train()

            self.save_progress()

    def save_progress(self) -> None:
        self.hyper_parameters.save_to_file(self.game)
        self.model.save_state()
        self.optimizer.save_state()

    def run(self) -> None:
        self.select_game()
        self.all_game_moves = self.game.generate_all_possible_moves()
        self.load_hyperparameters()
        self.load_model()
        self.model.share_memory()
        self.load_optimizer()
        self.load_mcts()
        mp.set_start_method("spawn", force=True)

        try:
            self.learn()
        finally:
            self.clear_training_folder()
