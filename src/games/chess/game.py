import typing as t
import numpy as np
from functools import cache
from numpy import typing as npt
from src import utils
from src.games import commons
from src.engines.alphazero.game import AlphaZeroGame
from src.games.chess import pieces as chess_pieces
from src.games.chess import utils as chess_utils
from src.games.chess.players import ChessPlayer
from src.games.chess.state import ChessState
from src.games.chess.commons import ChessPlayerType


class Chess(AlphaZeroGame):
    def setup(self) -> None:
        board: dict[str, chess_pieces.ChessPiece] = {}
        white = ChessPlayer(ChessPlayerType.WHITE)
        black = ChessPlayer(ChessPlayerType.BLACK)
        players = [white, black]

        # Add pawns
        for idx_col in range(8):
            for idx_row, player in [
                (6, white),
                (1, black),
            ]:
                pawn = chess_pieces.Pawn(player, (idx_row, idx_col))
                board[pawn.current_position_notation] = pawn
                player.add_piece(pawn)

        # Add other pieces
        for idx_row, player in [
            (7, white),
            (0, black),
        ]:
            piece_classes = [
                chess_pieces.Rook,
                chess_pieces.Knight,
                chess_pieces.Bishop,
                chess_pieces.Queen,
                chess_pieces.King,
                chess_pieces.Bishop,
                chess_pieces.Knight,
                chess_pieces.Rook,
            ]

            for idx_col in range(8):
                piece = piece_classes[idx_col](player, (idx_row, idx_col))
                board[piece.current_position_notation] = piece
                player.add_piece(piece)

        self._current_state = ChessState(board, players, 1)
        self._current_state.make_hash()
        self.update_state_squares_in_check(self._current_state)

    def update_state_squares_in_check(self, state: ChessState) -> None:
        for player in state.squares_in_check:
            state.squares_in_check[player] = set()

            for opponent in t.cast(list[ChessPlayer], state.players):
                if opponent == player:
                    continue

                for opponent_piece in opponent.pieces:
                    controlled_squares = opponent_piece.get_controlled_squares(state)
                    state.squares_in_check[player] |= controlled_squares

    @property
    def current_state(self) -> ChessState:
        return self._current_state

    def print_state(self, state: commons.GameState) -> None:
        state = t.cast(ChessState, state)
        print("Current game state:")

        for idx_row in range(8):
            row_text = []

            for idx_col in range(8):
                position_notation = chess_utils.position_to_notation[(idx_row, idx_col)]
                board_piece = state.board.get(position_notation)

                if board_piece:
                    row_text.append(str(board_piece))
                    continue

                if idx_row % 2 == idx_col % 2:
                    row_text.append("□")
                else:
                    row_text.append("■")

            print(" ".join(row_text))

        if state.winner:
            print(f'Player "{t.cast(ChessPlayer, state.winner).type.value}" wins')
        elif self.is_terminal(state):
            print("Draw")

    def _make_new_state(
        self,
        state: ChessState,
        current_player_idx: int,
        piece: chess_pieces.ChessPiece,
        next_position: tuple[int, int],
        promoted_piece_notation: t.Optional[str] = None,
        increase_moves: bool = True,
    ) -> ChessState:
        new_state = t.cast(ChessState, state.copy())
        new_state.current_player_idx = current_player_idx
        current_player = t.cast(ChessPlayer, new_state.current_player)
        new_state.move_count_no_progress += increase_moves
        new_state.move_count += increase_moves

        # Remove piece from the board
        old_idx_row, old_idx_col = piece.current_position
        new_state.board.pop(piece.current_position_notation)
        current_player.remove_piece(piece)

        # Make new piece
        new_piece = piece.copy()
        new_piece.current_position = next_position
        new_idx_row, new_idx_col = next_position
        new_position_notation = chess_utils.position_to_notation[next_position]

        # Check for promotion
        if promoted_piece_notation:
            match promoted_piece_notation:
                case chess_pieces.ChessPieceNotation.BISHOP.value:
                    new_piece = chess_pieces.Bishop(current_player, next_position)
                case chess_pieces.ChessPieceNotation.KNIGHT.value:
                    new_piece = chess_pieces.Knight(current_player, next_position)
                case chess_pieces.ChessPieceNotation.ROOK.value:
                    new_piece = chess_pieces.Rook(current_player, next_position)
                case chess_pieces.ChessPieceNotation.QUEEN.value:
                    new_piece = chess_pieces.Queen(current_player, next_position)

        match new_piece:
            case chess_pieces.Rook() | chess_pieces.King():
                new_piece.can_castle = False
            case chess_pieces.Pawn():
                new_piece.en_passant = abs(old_idx_row - new_idx_row) == 2

        # Check for capture
        captured_piece: t.Optional[chess_pieces.ChessPiece] = new_state.board.get(
            new_position_notation
        )

        if (
            not captured_piece
            and piece.notation == chess_pieces.ChessPieceNotation.PAWN
            and new_idx_col != old_idx_col
        ):  # Check for en-passant capture
            captured_position_notation = chess_utils.position_to_notation[
                (old_idx_row, new_idx_col)
            ]
            captured_piece = new_state.board.get(captured_position_notation)

            if not captured_piece:
                raise Exception("Invalid move. Expected en-passant capture")

            new_state.board.pop(captured_position_notation)

        if captured_piece:
            captured_piece.player.remove_piece(captured_piece)

        if piece.notation == chess_pieces.ChessPieceNotation.PAWN or captured_piece:
            new_state.move_count_no_progress = 0

        # Move piece
        new_state.board[new_position_notation] = new_piece
        current_player.add_piece(new_piece)

        new_state.make_hash()
        self.update_state_squares_in_check(new_state)
        return new_state

    @cache
    def evaluate_move(
        self, state: commons.GameState, move: str
    ) -> commons.Result[commons.GameState]:
        state = t.cast(ChessState, state)
        current_player_idx = state.get_next_player_idx()
        current_player = t.cast(ChessPlayer, state.players[current_player_idx])

        if move in {
            chess_pieces.Castling.KINGSIDE.value,
            chess_pieces.Castling.QUEENSIDE.value,
        }:
            king_moves = current_player.king.generate_possible_moves(state)

            if move not in king_moves:
                return commons.Result(error="Cannot castle")

            king_idx_row, king_idx_col = current_player.king.current_position

            if move == chess_pieces.Castling.KINGSIDE.value:
                rook_idx_col = 7
                new_king_idx_col = king_idx_col + 2
                new_rook_idx_col = new_king_idx_col - 1
            else:
                rook_idx_col = 0
                new_king_idx_col = king_idx_col - 2
                new_rook_idx_col = new_king_idx_col + 1

            rook_position_notation = chess_utils.position_to_notation[
                (king_idx_row, rook_idx_col)
            ]
            rook = state.board[rook_position_notation]
            new_state = self._make_new_state(
                state,
                current_player_idx,
                current_player.king,
                (king_idx_row, new_king_idx_col),
                increase_moves=False,
            )
            new_state = self._make_new_state(
                new_state,
                current_player_idx,
                rook,
                (king_idx_row, new_rook_idx_col),
            )
            return commons.Result(new_state)

        move_parts = move.split("=")

        if len(move_parts) > 2:
            return commons.Result(error="Invalid move")

        if len(move_parts[0]) not in {4, 5}:
            return commons.Result(error="Invalid move")

        piece_notation = ""
        current_position_notation, next_position_notation = (
            move_parts[0][-4:-2],
            move_parts[0][-2:],
        )

        if next_position_notation not in chess_utils.notation_to_position:
            return commons.Result(error="Invalid next position")

        next_position = chess_utils.notation_to_position[next_position_notation]

        if len(move_parts[0]) == 5:
            piece_notation = move_parts[0][0]

        promoted_piece_notation: t.Optional[str] = None

        if len(move_parts) == 2:
            promoted_piece_notation = move_parts[-1]

        piece = state.board.get(current_position_notation)

        if not piece or piece.notation.value != piece_notation:
            return commons.Result(error="Invalid piece/current position")

        piece_moves = piece.generate_possible_moves(state)

        if move not in piece_moves:
            return commons.Result(error="Invalid move")

        new_state = self._make_new_state(
            state,
            current_player_idx,
            piece,
            next_position,
            promoted_piece_notation,
        )

        if (
            current_player.king.current_position
            in new_state.squares_in_check[current_player]
        ):
            return commons.Result(error="King in check. Invalid move")

        return commons.Result(new_state)

    @cache
    def generate_possible_moves(
        self, state: commons.GameState
    ) -> list[tuple[str, commons.GameState]]:
        state = t.cast(ChessState, state)
        possible_moves: list[tuple[str, commons.GameState]] = []

        if state.winner:
            return possible_moves

        if state.move_count_no_progress >= 50:
            return possible_moves

        next_player = t.cast(ChessPlayer, state.players[state.get_next_player_idx()])
        pieces_possible_moves: list[str] = []

        for player_piece in next_player.pieces:
            piece_possible_moves = player_piece.generate_possible_moves(state)
            pieces_possible_moves.extend(piece_possible_moves)

        for piece_possible_move in pieces_possible_moves:
            result = self.evaluate_move(state, piece_possible_move)

            if result.success and result.data:
                possible_moves.append((piece_possible_move, result.data))

        if not possible_moves:
            # Check for check mate and assign winner. Otherwise it's a stalemate
            if next_player.king.current_position in state.squares_in_check[next_player]:
                state.winner = state.current_player

        possible_moves.sort(key=lambda move_state: move_state[0])
        return possible_moves

    def make_move(self) -> None:
        while True:
            next_player = t.cast(ChessPlayer, self.current_state.get_next_player())
            print(f'Player "{next_player.type.value}"\'s turn')
            possible_moves = self.generate_possible_moves(self.current_state)
            print("Choose one of the following possible moves:")

            for idx, possible_move in enumerate(possible_moves, start=1):
                print(f"{idx}. {possible_move[0]}")

            move = utils.prompt_input("Enter your move: ")
            result = self.evaluate_move(self.current_state, move)

            if not result.success and not result.data:
                print(result.error)
                continue

            self._current_state = t.cast(ChessState, result.data)
            break

    # AlphaZero methods
    @property
    def input_tensor_dimensions(self) -> tuple[int, int, int]:
        # The dimensions are, in order:
        # 1. 20 binary boards representing:
        #    1. White pawns
        #    2. White bishops
        #    3. White knights
        #    4. White rooks
        #    5. White queens (there might be more than one with promotion)
        #    6. White king
        #    7. White can castle kingside (only checks if the king and rook have not been moved, not if the king is in check)
        #    8. White can castle queenside (only checks if the king and rook have not been moved, not if the king is in check)
        #    9. Black pawns
        #    10. Black bishops
        #    11. Black knights
        #    12. Black rooks
        #    13. Black queens (there might be more than one with promotion)
        #    14. Black king
        #    15. Black can castle kingside (only checks if the king and rook have not been moved, not if the king is in check)
        #    16. Black can castle queenside (only checks if the king and rook have not been moved, not if the king is in check)
        #    17. Next player (the one that will change the current state)
        #    18. Board indicating on which square an en-passant capture is possible
        #    19. No progress move count (used in 50-move draw rule. Not binary like the others)
        #    20. Move count (used to force finish long matches on AlphaZero MCTS)
        # 2. Board height
        # 3. Board width
        return (20, 8, 8)

    @cache
    def generate_all_possible_moves(self) -> npt.NDArray[np.str_]:
        result: list[str] = []

        white = ChessPlayer(ChessPlayerType.WHITE)
        black = ChessPlayer(ChessPlayerType.BLACK)
        players = [white, black]
        state = ChessState({}, players, 1)
        # Pieces' moves are the same for whites and blacks, except pawns
        piece_classes_player = [
            (chess_pieces.Pawn, white),
            (chess_pieces.Pawn, black),
            (chess_pieces.Bishop, white),
            (chess_pieces.Knight, white),
            (chess_pieces.Rook, white),
            (chess_pieces.Queen, white),
            (chess_pieces.King, white),
        ]

        for piece_class, player in piece_classes_player:
            piece = piece_class(player, (0, 0))

            for idx_col in range(8):
                row_range = range(8)

                if isinstance(piece, chess_pieces.Pawn):
                    piece.force_capture = True
                    row_range = range(
                        piece.initial_row_idx,
                        piece.promotion_row_idx,
                        piece.direction,
                    )
                elif isinstance(piece, chess_pieces.King):
                    piece.can_castle = False

                for idx_row in row_range:
                    piece.current_position = (idx_row, idx_col)
                    state.board = {piece.current_position_notation: piece}
                    state.make_hash()  # Needs to remake hash because `piece.generate_possible_moves` is cached on state
                    possible_moves = piece.generate_possible_moves(state)
                    result.extend(possible_moves)

        result.extend(
            [
                chess_pieces.Castling.KINGSIDE.value,
                chess_pieces.Castling.QUEENSIDE.value,
            ]
        )
        result.sort()
        return np.array(result)

    ALPHAZERO_INPUT_CHANNEL_MAP = {
        "♙": 0,
        "♗": 1,
        "♘": 2,
        "♖": 3,
        "♕": 4,
        "♔": 5,
        "♟": 8,
        "♝": 9,
        "♞": 10,
        "♜": 11,
        "♛": 12,
        "♚": 13,
    }

    @cache
    def make_state_input_tensor(
        self, state: commons.GameState
    ) -> npt.NDArray[np.float32]:
        state = t.cast(ChessState, state)
        tensor = np.zeros(self.input_tensor_dimensions).astype(np.float32)
        en_passant_pawn: t.Optional[chess_pieces.Pawn] = None

        for piece in state.board.values():
            input_channel = self.ALPHAZERO_INPUT_CHANNEL_MAP[piece.symbol]
            idx_row, idx_col = piece.current_position
            tensor[input_channel, idx_row, idx_col] = 1

            if isinstance(piece, chess_pieces.Pawn) and piece.en_passant:
                en_passant_pawn = piece

        white = t.cast(ChessPlayer, state.players[0])
        black = t.cast(ChessPlayer, state.players[1])

        if white.king.can_castle_king_side(state):
            tensor[6] = 1

        if white.king.can_castle_queen_side(state):
            tensor[7] = 1

        if black.king.can_castle_king_side(state):
            tensor[14] = 1

        if black.king.can_castle_queen_side(state):
            tensor[15] = 1

        next_player = t.cast(ChessPlayer, state.get_next_player())

        if next_player == black:
            tensor[16] = 1

        if en_passant_pawn:
            idx_row, idx_col = en_passant_pawn.current_position
            tensor[17, idx_row, idx_col] = 1

        tensor[18] = state.move_count_no_progress
        tensor[19] = state.move_count
        return tensor

    def apply_state(self, state: commons.GameState) -> None:
        self._current_state = t.cast(ChessState, state)
