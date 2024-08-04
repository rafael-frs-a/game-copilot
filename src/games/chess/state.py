import typing as t
from src import utils
from src.games import commons
from src.games.chess.players import ChessPlayer
from src.games.chess import pieces as chess_pieces


class ChessState(commons.GameState):
    def __init__(
        self,
        board: dict[str, chess_pieces.ChessPiece],
        players: list[ChessPlayer],
        current_player_idx: int,
        winner: t.Optional[ChessPlayer] = None,
        moves_without_progress: int = 0,
        squares_in_check: t.Optional[dict[ChessPlayer, set[tuple[int, int]]]] = None,
    ) -> None:
        self.board = board
        self.players = t.cast(list[commons.Player], players)
        self.current_player_idx = current_player_idx
        self.winner = winner
        self.moves_without_progress = moves_without_progress

        if squares_in_check:
            self.squares_in_check = squares_in_check
        else:
            self.squares_in_check = {player: set() for player in players}

    def make_hash(self) -> None:
        pieces: list[str] = []

        for piece in self.board.values():
            pieces.append(f"{piece.symbol}{piece.current_position_notation}")

        state = "".join(sorted(pieces))
        state += t.cast(ChessPlayer, self.current_player).type.value
        state += str(self.moves_without_progress)
        self._hash = utils.make_hash_number(state)

    def copy(self) -> commons.GameState:
        board_copy: dict[str, chess_pieces.ChessPiece] = {}
        players_copy = [
            player.copy() for player in t.cast(list[ChessPlayer], self.players)
        ]

        for player in players_copy:
            for piece in player.pieces:
                board_copy[piece.current_position_notation] = piece

        squares_in_check = {
            player: squares for player, squares in self.squares_in_check.items()
        }

        return ChessState(
            board_copy,
            players_copy,
            self.current_player_idx,
            t.cast(
                t.Optional[ChessPlayer], self.winner
            ),  # No need to copy the winner, since it's used to determine terminal states
            self.moves_without_progress,
            squares_in_check,
        )
