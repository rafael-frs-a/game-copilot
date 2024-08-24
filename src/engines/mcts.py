import math
import typing as t
import random
from src import utils
from src.games import commons as games_commons
from . import commons


class Node:
    def __init__(
        self, move: str, state: games_commons.GameState, parent: t.Optional["Node"]
    ) -> None:
        self.move = move  # Input command that leads to the informed state
        self.state = state
        self.parent = parent
        self.children: list[Node] = []
        self.visits = 0
        self.reward = 0.0
        self.wins = 0  # Not actually used by MCTS. Just to display winning chances

    def add_child(self, move: str, state: games_commons.GameState) -> "Node":
        child_node = Node(move, state, self)
        self.children.append(child_node)
        return child_node

    def update(self, winner: t.Optional[games_commons.Player]) -> None:
        self.visits += 1
        reward = 0.5  # tie/draw

        if winner:
            reward = 1.0 if winner == self.state.current_player else 0.0

        self.reward += reward
        self.wins += winner == self.state.current_player


class MCTS(commons.Engine):
    def set_seed(self) -> None:
        # Ask for numeric seed
        # This allows deterministic reproducibility
        # Inform nothing if we want to skip it
        while True:
            seed = utils.prompt_input("Enter a numeric seed (optional): ")

            if seed == "":
                break

            try:
                self.seed = int(seed)
                random.seed(self.seed)
                break
            except ValueError:
                print(
                    "Invalid input. Please enter a valid number or press enter to skip this step"
                )

    def set_iterations(self) -> None:
        # Number of iterations needed for MCTS
        while True:
            iterations = utils.prompt_input("Enter the number of iterations: ")

            try:
                self.iterations = int(iterations)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")

    def set_ucb_constant(self) -> None:
        self.ucb_constant = math.sqrt(2)

        while True:
            ucb_constant = utils.prompt_input(
                "Enter the UCB constant (optional, default to √2): "
            )

            if not ucb_constant:
                break

            try:
                self.ucb_constant = float(ucb_constant)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")

    def set_max_game_moves(self) -> None:
        self.max_game_moves = float("inf")

        while True:
            max_game_moves = utils.prompt_input(
                "Enter the max. number of moves on game simulations (optional): "
            )

            if not max_game_moves:
                break

            try:
                self.max_game_moves = float(int(max_game_moves))
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")

    def setup(self) -> None:
        self.set_seed()
        self.set_iterations()
        self.set_ucb_constant()
        self.set_max_game_moves()

    def ucb_select(self, parent_node: Node) -> Node:
        def ucb_value(child_node: Node) -> float:
            if child_node.visits == 0:
                return float("inf")

            exploitation = child_node.reward / child_node.visits
            exploration = self.ucb_constant * math.sqrt(
                math.log(parent_node.visits) / child_node.visits
            )
            return exploitation + exploration

        return max(
            parent_node.children,
            key=ucb_value,
        )

    def expand(self, node: Node) -> Node:
        if not node.children:
            possible_moves = self.game.generate_possible_moves(node.state)

            for move, state in possible_moves:
                node.add_child(move, state)

        return random.choice(node.children)

    def simulate(self, node: Node) -> t.Optional[games_commons.Player]:
        state = node.state
        n_moves = 0

        while not self.game.is_terminal(state) and n_moves < self.max_game_moves:
            possible_moves = self.game.generate_possible_moves(state)
            _, state = random.choice(possible_moves)
            n_moves += 1

        return state.winner

    def backpropagate(
        self, node: Node, winner: t.Optional[games_commons.Player]
    ) -> None:
        current_node: t.Optional[Node] = node

        while current_node:
            current_node.update(winner)
            current_node = current_node.parent

    def suggest_move(self) -> None:
        root_node = Node("", self.game.current_state, None)

        for _ in range(self.iterations):
            node = root_node

            while node.children:
                node = self.ucb_select(node)

            if not self.game.is_terminal(node.state):
                # We want to simulate a game for every node before expanding it, except the root node
                if node.visits > 0 or node == root_node:
                    node = self.expand(node)

            winner = self.simulate(node)
            self.backpropagate(node, winner)

        default_child = Node("None", root_node.state, root_node)
        best_child = max(
            root_node.children,
            key=lambda child_node: child_node.visits,
            default=default_child,  # Used in case the number of iterations is zero
        )
        chances = (best_child.wins / best_child.visits) if best_child.visits > 0 else 0
        print(f"Suggested move: {best_child.move}. Winning chances: {chances}")
