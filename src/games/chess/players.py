import typing as t
from src import utils
from src.games import commons
from src.games.chess import pieces as chess_pieces
from src.games.chess.commons import ChessPlayerType


class ChessPlayer(commons.Player):
    def __init__(self, type_: ChessPlayerType) -> None:
        self.type = type_
        self._pieces: set[chess_pieces.ChessPiece] = set()
        self._king: t.Optional[chess_pieces.King] = None
        self.make_hash()

    def make_hash(self) -> None:
        self._hash = utils.make_hash_number(self.type.value)

    def __hash__(self) -> int:
        return self._hash

    def add_piece(self, piece: chess_pieces.ChessPiece) -> None:
        self._pieces.add(piece)

        if isinstance(piece, chess_pieces.King):
            self._king = piece

    def remove_piece(self, piece: chess_pieces.ChessPiece) -> None:
        self._pieces.remove(piece)

    @property
    def king(self) -> chess_pieces.King:
        if not self._king:
            raise Exception("King not assigned")

        return self._king

    @property
    def pieces(self) -> set[chess_pieces.ChessPiece]:
        return self._pieces

    def copy(self) -> "ChessPlayer":
        player = ChessPlayer(self.type)

        for piece in self.pieces:
            piece_copy = piece.copy()
            piece_copy.player = player
            player.add_piece(piece_copy)

        return player
