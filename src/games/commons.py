import abc
import typing as t
from functools import cache


class Player:
    @abc.abstractmethod
    def make_hash(self) -> None:
        pass

    def __hash__(self) -> int:
        if not hasattr(self, "_hash"):
            raise Exception("Hash not defined")

        return t.cast(int, self._hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Player):
            return False

        return self.__hash__() == other.__hash__()


class GameState(abc.ABC):
    # Definition of current player: the player whose action leads to this state at the end of their turn
    players: list[Player]
    current_player_idx: int
    winner: t.Optional[Player]

    @abc.abstractmethod
    def make_hash(self) -> None:
        pass

    def __hash__(self) -> int:
        if not hasattr(self, "_hash"):
            raise Exception("Hash not defined")

        return t.cast(int, self._hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return False

        return self.__hash__() == other.__hash__()

    @abc.abstractmethod
    def copy(self) -> "GameState":
        pass

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_idx]

    def get_next_player_idx(self) -> int:
        return (self.current_player_idx + 1) % len(self.players)

    def get_next_player(self) -> Player:
        return self.players[self.get_next_player_idx()]


TData = t.TypeVar("TData")


class Result(t.Generic[TData]):
    def __init__(
        self, data: t.Optional[TData] = None, error: t.Optional[str] = None
    ) -> None:
        self.data = data
        self.error = error

    @property
    def success(self) -> bool:
        return not self.error


class Game(abc.ABC):
    @abc.abstractmethod
    def setup(self) -> None:
        pass

    @property
    @abc.abstractmethod
    def current_state(self) -> GameState:
        pass

    @cache
    @abc.abstractmethod
    def evaluate_move(self, state: GameState, move: str) -> Result[GameState]:
        pass

    @cache
    @abc.abstractmethod
    # List of tuples containing the "action" and the "state" it leads to
    def generate_possible_moves(self, state: GameState) -> list[tuple[str, GameState]]:
        pass

    def is_terminal(self, state: GameState) -> bool:
        return len(self.generate_possible_moves(state)) == 0

    @abc.abstractmethod
    def print_state(self, state: GameState) -> None:
        pass

    @abc.abstractmethod
    def make_move(self) -> None:
        pass
