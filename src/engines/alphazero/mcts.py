import math
import typing as t
import torch
import numpy as np
from numpy import typing as npt
from src.games import commons as games_commons
from src.engines.alphazero.game import AlphaZeroGame
from src.engines.alphazero.hyper_parameters import HyperParameters
from src.engines.alphazero.model import GameNet
from . import utils


class Node:
    def __init__(
        self,
        move_idx: int,
        state: games_commons.GameState,
        parent: t.Optional["Node"],
        prior_prob: float = 0.0,
    ) -> None:
        # Index of the input command in the list of all actions that leads to the informed state
        self.move_idx = move_idx
        self.state = state
        self.parent = parent
        self.children: list[Node] = []
        self.prior_prob = prior_prob
        self.visits = 0
        self.reward = 0.0

    def add_child(
        self, move_idx: int, state: games_commons.GameState, prior_prob: float
    ) -> "Node":
        child_node = Node(move_idx, state, self, prior_prob)
        self.children.append(child_node)
        return child_node

    def update(self, reward: float) -> None:
        self.visits += 1
        self.reward += reward


class MCTS:
    def __init__(
        self, game: AlphaZeroGame, hyper_parameters: HyperParameters, model: GameNet
    ) -> None:
        self.game = game
        self.hyper_parameters = hyper_parameters
        self.model = model
        self.all_game_moves = self.game.generate_all_possible_moves()

    def ucb_select(self, parent_node: Node) -> Node:
        def ucb_value(child_node: Node) -> float:
            q_value = 0.0

            if child_node.visits > 0:
                q_value = 1 - ((child_node.reward / child_node.visits) + 1) / 2

            exploration = (
                self.hyper_parameters.mcts_puct_constant
                * (math.sqrt(parent_node.visits) / (child_node.visits + 1))
                * child_node.prior_prob
            )
            return q_value + exploration

        return max(
            parent_node.children,
            key=ucb_value,
        )

    def expand(self, node: Node) -> tuple[npt.NDArray[np.float32], float]:
        input_array = self.game.make_state_input_tensor(node.state)
        input_tensor = utils.make_torch_tensor(input_array)
        input_tensor = input_tensor.unsqueeze(0)
        policy, value = t.cast(
            tuple[torch.Tensor, torch.Tensor], self.model(input_tensor)
        )
        policy_array = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()

        if node.parent is None:
            # Add Dirichlet noise to the root node
            policy_array = (
                1 - self.hyper_parameters.dirichlet_noise_epsilon
            ) * policy_array + self.hyper_parameters.dirichlet_noise_epsilon * np.random.dirichlet(
                [self.hyper_parameters.dirichlet_noise_alpha] * len(self.all_game_moves)
            )

        valid_moves = self.game.generate_possible_moves(node.state)
        valid_moves_map: dict[str, games_commons.GameState] = dict(valid_moves)
        valid_moves_array = np.array([valid_move for valid_move in valid_moves_map])
        valid_mask = np.isin(self.all_game_moves, valid_moves_array)
        policy_array[~valid_mask] = 0  # Zero out invalid moves
        policy_array /= np.sum(
            policy_array
        )  # Recalculate prob. distribution after zeroing out invalid moves

        for move_idx, prob in enumerate(policy_array):
            if prob <= 0:
                continue

            move = self.all_game_moves[move_idx]
            state = valid_moves_map[move]
            node.add_child(move_idx, state, prob)

        return policy_array, value.item()

    def backpropagate(
        self,
        node: Node,
        winner: t.Optional[games_commons.Player],
        value: t.Optional[float],
    ) -> None:
        current_node: t.Optional[Node] = node
        original_player = node.state.current_player

        if value is None:
            # Since the MCTS is not simulating games to a terminal state
            # it should never happen that the winner, if there is one,
            # is different from the node's current player, ie a defeat, only win or draw.
            value = 1.0 if winner else 0.0

            # However, we need to get the value from the perspective of the previous player,
            # the parent node's state, since, in AlphaZero MCTS, the value from a node's state
            # is used to update its children, not itself

            # Should always have a parent. Just doing this so mypy doesn't complain
            if node.parent:
                value = self.game.get_state_value(
                    node.parent.state, original_player, value
                )

        while current_node:
            current_node.update(
                self.game.get_state_value(current_node.state, original_player, value)
            )
            current_node = current_node.parent

    @torch.no_grad()
    def search(self) -> tuple[npt.NDArray[np.float32], float]:
        root_node = Node(-1, self.game.current_state, None)
        root_policy, root_value = self.expand(root_node)

        if self.hyper_parameters.mcts_num_iterations == 0:
            # This could be used when the model is trained and we want to test playing it without MCTS
            return root_policy, root_value

        for _ in range(self.hyper_parameters.mcts_num_iterations):
            node = root_node

            while node.children:
                node = self.ucb_select(node)

            is_terminal = self.game.is_terminal(node.state)
            winner = node.state.winner
            value: t.Optional[float] = None

            if not is_terminal:
                _, value = self.expand(node)

            self.backpropagate(node, winner, value)

        move_probs = np.zeros(len(self.all_game_moves)).astype(np.float32)

        for child in root_node.children:
            move_probs[child.move_idx] = child.visits

        move_probs /= np.sum(move_probs)
        return move_probs, root_value
