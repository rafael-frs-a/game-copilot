from enum import Enum
from src.games.commons import Game
from src.engines.commons import Engine
from src.engines.mcts import MCTS
from src.engines.alphazero import AlphaZero


class EngineType(Enum):
    MCTS = "mcts"
    ALPHA_ZERO = "alphazero"


def make_engine(engine_type: str, game: Game) -> Engine:
    if engine_type == EngineType.MCTS.value:
        return MCTS(game)
    elif engine_type == EngineType.ALPHA_ZERO.value:
        return AlphaZero(game)

    raise ValueError("The selected engine is not supported")
