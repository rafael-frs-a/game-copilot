from enum import Enum
from src.games.commons import Game
from src.engines.commons import Engine
from src.engines.mcts import MCTS


class EngineType(Enum):
    MCTS = "mcts"


def make_engine(engine_type: str, game: Game) -> Engine:
    if engine_type == EngineType.MCTS.value:
        return MCTS(game)

    raise ValueError("The selected engine is not supported")
