import abc
import typing as t
from enum import Enum
from functools import cache, lru_cache
from src import utils
from src.games.chess import constants
from src.games.chess import utils as chess_utils
from src.games.chess.commons import ChessPlayerType


if t.TYPE_CHECKING:
    from src.games.chess.state import ChessState
    from src.games.chess.players import ChessPlayer


class ChessPieceNotation(Enum):
    PAWN = ""
    BISHOP = "B"
    KNIGHT = "N"
    ROOK = "R"
    QUEEN = "Q"
    KING = "K"


class Castling(Enum):
    KINGSIDE = "O-O"
    QUEENSIDE = "O-O-O"


ALL_CHESS_PIECES_NOTATIONS = {
    ChessPieceNotation.PAWN.value,
    ChessPieceNotation.BISHOP.value,
    ChessPieceNotation.KNIGHT.value,
    ChessPieceNotation.ROOK.value,
    ChessPieceNotation.QUEEN.value,
    ChessPieceNotation.KING.value,
}
CHESS_PIECES_NOTATIONS_PROMOTION = {
    ChessPieceNotation.BISHOP.value,
    ChessPieceNotation.KNIGHT.value,
    ChessPieceNotation.ROOK.value,
    ChessPieceNotation.QUEEN.value,
}


class ChessPiece(abc.ABC):
    notation: ChessPieceNotation
    symbol: str

    def __init__(
        self, player: "ChessPlayer", current_position: tuple[int, int]
    ) -> None:
        self.player = player
        self.current_position = current_position

    @property
    def current_position(self) -> tuple[int, int]:
        return self._current_position

    @current_position.setter
    def current_position(self, value: tuple[int, int]) -> None:
        self._current_position = value
        self._current_position_notation = chess_utils.position_to_notation[value]
        self.make_hash()

    @property
    def current_position_notation(self) -> str:
        return self._current_position_notation

    @current_position_notation.setter
    def current_position_notation(self, value: str) -> None:
        self._current_position_notation = value
        self._current_position = chess_utils.notation_to_position[value]
        self.make_hash()

    def __str__(self) -> str:
        return self.symbol

    def make_hash(self) -> None:
        if not hasattr(self, "symbol"):
            return

        state = self.symbol + self.current_position_notation
        self._hash = utils.make_hash_number(state)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChessPiece):
            return False

        return self.__hash__() == other.__hash__()

    @cache
    @abc.abstractmethod
    def get_controlled_squares(self, state: "ChessState") -> set[tuple[int, int]]:
        pass

    @cache
    @abc.abstractmethod
    def generate_possible_moves(self, state: "ChessState") -> set[str]:
        pass

    def make_piece_move_command(self, next_position: tuple[int, int]) -> str:
        move = self.notation.value
        move += self.current_position_notation
        move += chess_utils.position_to_notation[next_position]
        return move

    @abc.abstractmethod
    def copy(self) -> "ChessPiece":
        pass


class Pawn(ChessPiece):
    def __init__(
        self,
        player: "ChessPlayer",
        current_position: tuple[int, int],
        en_passant: bool = False,
        force_capture: bool = False,
    ) -> None:
        super().__init__(player, current_position)
        self.en_passant = en_passant
        self.force_capture = (
            force_capture  # Used when calculating all possible moves for AlphaZero
        )
        self.notation = ChessPieceNotation.PAWN
        self.symbol = "♙" if self.player.type == ChessPlayerType.WHITE else "♟"
        self.direction = -1 if self.player.type == ChessPlayerType.WHITE else 1
        self.initial_row_idx = 6 if self.player.type == ChessPlayerType.WHITE else 1
        self.promotion_row_idx = 0 if self.player.type == ChessPlayerType.WHITE else 7
        self.make_hash()

    @cache
    def _get_controlled_squares(
        self, current_position: tuple[int, int]
    ) -> set[tuple[int, int]]:
        controlled_squares: set[tuple[int, int]] = set()
        idx_row, idx_col = current_position
        capture_row = idx_row + self.direction
        capture_columns = [idx_col - 1, idx_col + 1]

        # Should not happen because of promotion
        if not 0 <= capture_row < 8:
            raise Exception("Invalid board position")

        for capture_column in capture_columns:
            if 0 <= capture_column < 8:
                controlled_squares.add((capture_row, capture_column))

        return controlled_squares

    @lru_cache(maxsize=constants.CACHE_MAX_SIZE)
    def get_controlled_squares(self, state: "ChessState") -> set[tuple[int, int]]:
        # The state doesn't actually matter when determining the squares controlled by the pawn
        # since its capture range is of just one square
        return self._get_controlled_squares(self.current_position)

    def _make_piece_move_commands(
        self, next_idx_row: int, next_idx_col: int
    ) -> set[str]:
        result: set[str] = set()
        last_idx_row = 0 if self.player.type == ChessPlayerType.WHITE else 7
        base_move = self.make_piece_move_command((next_idx_row, next_idx_col))

        # Promotion
        if next_idx_row == last_idx_row:
            for piece_class in CHESS_PIECES_NOTATIONS_PROMOTION:
                result.add(f"{base_move}={piece_class}")

            return result

        result.add(base_move)
        return result

    @lru_cache(maxsize=constants.CACHE_MAX_SIZE)
    def generate_possible_moves(self, state: "ChessState") -> set[str]:
        possible_moves: set[str] = set()
        idx_row, idx_col = self.current_position
        move_range = 2 if idx_row == self.initial_row_idx else 1

        for _ in range(move_range):
            idx_row += self.direction

            # Should not happen because of promotion
            if not 0 <= idx_row < 8:
                raise Exception("Invalid board position")

            position_notation = chess_utils.position_to_notation[(idx_row, idx_col)]

            if position_notation in state.board:
                break

            possible_moves |= self._make_piece_move_commands(idx_row, idx_col)

        controlled_squares = self._get_controlled_squares(self.current_position)

        def can_capture(idx_row: int, idx_col: int) -> bool:
            if self.force_capture:
                return True

            position_notation = chess_utils.position_to_notation[(idx_row, idx_col)]
            board_piece = state.board.get(position_notation)

            if board_piece:
                if board_piece.player == self.player:
                    return False

                if board_piece.notation == ChessPieceNotation.KING:
                    raise Exception("Invalid board position. Cannot capture the king")

                return True

            # Check for en-passant
            position_notation = chess_utils.position_to_notation[
                (self.current_position[0], idx_col)
            ]
            board_piece = state.board.get(position_notation)

            if not board_piece:
                return False

            if board_piece.player == self.player or not isinstance(board_piece, Pawn):
                return False

            return board_piece.en_passant

        for idx_row, idx_col in controlled_squares:
            if can_capture(idx_row, idx_col):
                possible_moves |= self._make_piece_move_commands(idx_row, idx_col)

        return possible_moves

    def copy(self) -> ChessPiece:
        return Pawn(
            self.player, self.current_position, False
        )  # en-passant right lasts for only one turn


class Bishop(ChessPiece):
    def __init__(
        self, player: "ChessPlayer", current_position: tuple[int, int]
    ) -> None:
        super().__init__(player, current_position)
        self.notation = ChessPieceNotation.BISHOP
        self.symbol = "♗" if self.player.type == ChessPlayerType.WHITE else "♝"
        self.make_hash()

    @lru_cache(maxsize=constants.CACHE_MAX_SIZE)
    def get_controlled_squares(self, state: "ChessState") -> set[tuple[int, int]]:
        controlled_squares: set[tuple[int, int]] = set()
        actions = [
            (-1, -1),  # Top-left diagonal
            (-1, 1),  # Top-right diagonal
            (1, -1),  # Bottom-left diagonal
            (1, 1),  # Bottom-right diagonal
        ]

        for add_row, add_col in actions:
            idx_row, idx_col = self.current_position
            idx_row += add_row
            idx_col += add_col

            while 0 <= idx_row < 8 and 0 <= idx_col < 8:
                controlled_squares.add((idx_row, idx_col))
                position_notation = chess_utils.position_to_notation[(idx_row, idx_col)]

                if position_notation in state.board:
                    break

                idx_row += add_row
                idx_col += add_col

        return controlled_squares

    @lru_cache(maxsize=constants.CACHE_MAX_SIZE)
    def generate_possible_moves(self, state: "ChessState") -> set[str]:
        possible_moves: set[str] = set()
        controlled_squares = self.get_controlled_squares(state)

        for idx_row, idx_col in controlled_squares:
            position_notation = chess_utils.position_to_notation[(idx_row, idx_col)]
            board_piece = state.board.get(position_notation)

            if not board_piece:
                possible_moves.add(self.make_piece_move_command((idx_row, idx_col)))
                continue

            if board_piece.player != self.player:
                # Should not happen
                if board_piece.notation == ChessPieceNotation.KING:
                    raise Exception("Invalid board position. Cannot capture the king")

                possible_moves.add(self.make_piece_move_command((idx_row, idx_col)))

        return possible_moves

    def copy(self) -> ChessPiece:
        return Bishop(self.player, self.current_position)


class Knight(ChessPiece):
    def __init__(
        self, player: "ChessPlayer", current_position: tuple[int, int]
    ) -> None:
        super().__init__(player, current_position)
        self.notation = ChessPieceNotation.KNIGHT
        self.symbol = "♘" if self.player.type == ChessPlayerType.WHITE else "♞"
        self.make_hash()

    @cache
    def _get_controlled_squares(
        self, current_position: tuple[int, int]
    ) -> set[tuple[int, int]]:
        controlled_squares: set[tuple[int, int]] = set()
        # Possible "L" configurations
        actions = [
            (-2, -1),
            (-1, -2),
            (-2, 1),
            (-1, 2),
            (2, -1),
            (1, -2),
            (2, 1),
            (1, 2),
        ]

        for add_row, add_col in actions:
            idx_row, idx_col = current_position
            idx_row += add_row
            idx_col += add_col

            if 0 <= idx_row < 8 and 0 <= idx_col < 8:
                controlled_squares.add((idx_row, idx_col))

        return controlled_squares

    @lru_cache(maxsize=constants.CACHE_MAX_SIZE)
    def get_controlled_squares(self, state: "ChessState") -> set[tuple[int, int]]:
        # The state doesn't actually matter when determining the squares controlled by the knight
        # since its capture range is not limited by pieces in the way
        return self._get_controlled_squares(self.current_position)

    @lru_cache(maxsize=constants.CACHE_MAX_SIZE)
    def generate_possible_moves(self, state: "ChessState") -> set[str]:
        possible_moves: set[str] = set()
        controlled_squares = self._get_controlled_squares(self.current_position)

        for idx_row, idx_col in controlled_squares:
            position_notation = chess_utils.position_to_notation[(idx_row, idx_col)]
            board_piece = state.board.get(position_notation)

            if not board_piece:
                possible_moves.add(self.make_piece_move_command((idx_row, idx_col)))
                continue

            if board_piece.player != self.player:
                # Should not happen
                if board_piece.notation == ChessPieceNotation.KING:
                    raise Exception("Invalid board position. Cannot capture the king")

                possible_moves.add(self.make_piece_move_command((idx_row, idx_col)))

        return possible_moves

    def copy(self) -> ChessPiece:
        return Knight(self.player, self.current_position)


class Rook(ChessPiece):
    def __init__(
        self,
        player: "ChessPlayer",
        current_position: tuple[int, int],
        can_castle: bool = True,
    ) -> None:
        super().__init__(player, current_position)
        self.can_castle = can_castle
        self.notation = ChessPieceNotation.ROOK
        self.symbol = "♖" if self.player.type == ChessPlayerType.WHITE else "♜"
        self.make_hash()

    @lru_cache(maxsize=constants.CACHE_MAX_SIZE)
    def get_controlled_squares(self, state: "ChessState") -> set[tuple[int, int]]:
        controlled_squares: set[tuple[int, int]] = set()
        actions = [
            (-1, 0),  # Up
            (0, -1),  # Left
            (1, 0),  # Down
            (0, 1),  # Right
        ]

        for add_row, add_col in actions:
            idx_row, idx_col = self.current_position
            idx_row += add_row
            idx_col += add_col

            while 0 <= idx_row < 8 and 0 <= idx_col < 8:
                controlled_squares.add((idx_row, idx_col))
                position_notation = chess_utils.position_to_notation[(idx_row, idx_col)]

                if position_notation in state.board:
                    break

                idx_row += add_row
                idx_col += add_col

        return controlled_squares

    @lru_cache(maxsize=constants.CACHE_MAX_SIZE)
    def generate_possible_moves(self, state: "ChessState") -> set[str]:
        possible_moves: set[str] = set()
        controlled_squares = self.get_controlled_squares(state)

        for idx_row, idx_col in controlled_squares:
            position_notation = chess_utils.position_to_notation[(idx_row, idx_col)]
            board_piece = state.board.get(position_notation)

            if not board_piece:
                possible_moves.add(self.make_piece_move_command((idx_row, idx_col)))
                continue

            if board_piece.player != self.player:
                # Should not happen
                if board_piece.notation == ChessPieceNotation.KING:
                    raise Exception("Invalid board position. Cannot capture the king")

                possible_moves.add(self.make_piece_move_command((idx_row, idx_col)))

        return possible_moves

    def copy(self) -> ChessPiece:
        return Rook(self.player, self.current_position, self.can_castle)


class Queen(ChessPiece):
    def __init__(
        self, player: "ChessPlayer", current_position: tuple[int, int]
    ) -> None:
        super().__init__(player, current_position)
        self.notation = ChessPieceNotation.QUEEN
        self.symbol = "♕" if self.player.type == ChessPlayerType.WHITE else "♛"
        self.make_hash()

    @lru_cache(maxsize=constants.CACHE_MAX_SIZE)
    def get_controlled_squares(self, state: "ChessState") -> set[tuple[int, int]]:
        controlled_squares: set[tuple[int, int]] = set()
        actions = [
            (-1, 0),  # Up
            (0, -1),  # Left
            (1, 0),  # Down
            (0, 1),  # Right
            (-1, -1),  # Top-left diagonal
            (-1, 1),  # Top-right diagonal
            (1, -1),  # Bottom-left diagonal
            (1, 1),  # Bottom-right diagonal
        ]

        for add_row, add_col in actions:
            idx_row, idx_col = self.current_position
            idx_row += add_row
            idx_col += add_col

            while 0 <= idx_row < 8 and 0 <= idx_col < 8:
                controlled_squares.add((idx_row, idx_col))
                position_notation = chess_utils.position_to_notation[(idx_row, idx_col)]

                if position_notation in state.board:
                    break

                idx_row += add_row
                idx_col += add_col

        return controlled_squares

    @lru_cache(maxsize=constants.CACHE_MAX_SIZE)
    def generate_possible_moves(self, state: "ChessState") -> set[str]:
        possible_moves: set[str] = set()
        controlled_squares = self.get_controlled_squares(state)

        for idx_row, idx_col in controlled_squares:
            position_notation = chess_utils.position_to_notation[(idx_row, idx_col)]
            board_piece = state.board.get(position_notation)

            if not board_piece:
                possible_moves.add(self.make_piece_move_command((idx_row, idx_col)))
                continue

            if board_piece.player != self.player:
                # Should not happen
                if board_piece.notation == ChessPieceNotation.KING:
                    raise Exception("Invalid board position. Cannot capture the king")

                possible_moves.add(self.make_piece_move_command((idx_row, idx_col)))

        return possible_moves

    def copy(self) -> ChessPiece:
        return Queen(self.player, self.current_position)


class King(ChessPiece):
    def __init__(
        self,
        player: "ChessPlayer",
        current_position: tuple[int, int],
        can_castle: bool = True,
    ) -> None:
        super().__init__(player, current_position)
        self.can_castle = can_castle
        self.notation = ChessPieceNotation.KING
        self.symbol = "♔" if self.player.type == ChessPlayerType.WHITE else "♚"
        self.make_hash()

    @cache
    def _get_controlled_squares(
        self, current_position: tuple[int, int]
    ) -> set[tuple[int, int]]:
        controlled_squares: set[tuple[int, int]] = set()
        actions = [
            (-1, 0),  # Up
            (0, -1),  # Left
            (1, 0),  # Down
            (0, 1),  # Right
            (-1, -1),  # Top-left diagonal
            (-1, 1),  # Top-right diagonal
            (1, -1),  # Bottom-left diagonal
            (1, 1),  # Bottom-right diagonal
        ]

        for add_row, add_col in actions:
            idx_row, idx_col = current_position
            idx_row += add_row
            idx_col += add_col

            if 0 <= idx_row < 8 and 0 <= idx_col < 8:
                controlled_squares.add((idx_row, idx_col))

        return controlled_squares

    @lru_cache(maxsize=constants.CACHE_MAX_SIZE)
    def get_controlled_squares(self, state: "ChessState") -> set[tuple[int, int]]:
        # The state doesn't actually matter when determining the squares controlled by the king
        # since its capture range is of just one square
        return self._get_controlled_squares(self.current_position)

    def _can_castle(
        self,
        state: "ChessState",
        add_col: int,
        add_col_direction: int,
    ) -> bool:
        if not self.can_castle:
            return False

        current_idx_row, current_idx_col = self.current_position
        position_notation = chess_utils.position_to_notation[
            (current_idx_row, current_idx_col + add_col * add_col_direction)
        ]
        rook = state.board.get(position_notation)

        if not rook:
            return False

        if not isinstance(rook, Rook):
            return False

        if not rook.can_castle:
            return False

        # Where the king will pass through, including where it is. Must not be a place in check
        for idx_col in range(
            current_idx_col, current_idx_col + 3 * add_col_direction, add_col_direction
        ):
            if (current_idx_row, idx_col) in state.squares_in_check[self.player]:
                return False

        # Space between king and rook. Must be not occupied
        for idx_col in range(
            current_idx_col + 1 * add_col_direction,
            current_idx_col + add_col * add_col_direction,
            add_col_direction,
        ):
            position_notation = chess_utils.position_to_notation[
                (current_idx_row, idx_col)
            ]

            if position_notation in state.board:
                return False

        return True

    def can_castle_king_side(self, state: "ChessState") -> bool:
        return self._can_castle(state, 3, 1)

    def can_castle_queen_side(self, state: "ChessState") -> bool:
        return self._can_castle(state, 4, -1)

    @lru_cache(maxsize=constants.CACHE_MAX_SIZE)
    def generate_possible_moves(self, state: "ChessState") -> set[str]:
        possible_moves: set[str] = set()
        controlled_squares = self._get_controlled_squares(self.current_position)

        for idx_row, idx_col in controlled_squares:
            # Cannot move to a square In check
            if (idx_row, idx_col) in state.squares_in_check[self.player]:
                continue

            position_notation = chess_utils.position_to_notation[(idx_row, idx_col)]
            board_piece = state.board.get(position_notation)

            if not board_piece:
                possible_moves.add(self.make_piece_move_command((idx_row, idx_col)))
                continue

            if board_piece.player != self.player:
                # Should not happen
                if board_piece.notation == ChessPieceNotation.KING:
                    raise Exception("Invalid board position. Cannot capture the king")

                possible_moves.add(self.make_piece_move_command((idx_row, idx_col)))

        if self.can_castle_king_side(state):
            possible_moves.add(Castling.KINGSIDE.value)

        if self.can_castle_queen_side(state):
            possible_moves.add(Castling.QUEENSIDE.value)

        return possible_moves

    def copy(self) -> ChessPiece:
        return King(self.player, self.current_position, self.can_castle)
