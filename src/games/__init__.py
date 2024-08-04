from enum import Enum
from src.games.commons import Game
from src.games.tic_tac_toe import TicTacToe
from src.games.chess import Chess


class GameType(Enum):
    TIC_TAC_TOE = "tic-tac-toe"
    CHESS = "chess"


def make_game(game_type: str) -> Game:
    if game_type == GameType.TIC_TAC_TOE.value:
        return TicTacToe()

    if game_type == GameType.CHESS.value:
        return Chess()

    raise ValueError("The selected game is not supported")
