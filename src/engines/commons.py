import abc
from src.games.commons import Game


class Engine(abc.ABC):
    def __init__(self, game: Game) -> None:
        self.game = game

    @abc.abstractmethod
    def setup(self) -> None:
        pass

    @abc.abstractmethod
    def suggest_move(self) -> None:
        pass
