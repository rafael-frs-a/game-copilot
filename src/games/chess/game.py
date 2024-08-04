import typing as t
from functools import cache
from src import utils
from src.games import commons
from src.games.chess import pieces as chess_pieces
from src.games.chess import utils as chess_utils
from src.games.chess.players import ChessPlayer
from src.games.chess.state import ChessState
from src.games.chess.commons import ChessPlayerType


class Chess(commons.Game):
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
                piece = piece_classes[idx_col](
                    player, (idx_row, idx_col)
                )  # type: ignore[abstract]
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
            print(f'Player "{t.cast(ChessPlayer, state.winner).type}" wins')
        elif self.is_terminal(state):
            print("Draw")

    def _make_new_state(
        self,
        state: ChessState,
        current_player_idx: int,
        piece: chess_pieces.ChessPiece,
        next_position: tuple[int, int],
        promoted_piece_notation: t.Optional[str] = None,
        increase_moves_without_progress: bool = True,
    ) -> ChessState:
        new_state = t.cast(ChessState, state.copy())
        new_state.current_player_idx = current_player_idx
        current_player = t.cast(ChessPlayer, new_state.current_player)
        new_state.moves_without_progress += increase_moves_without_progress

        # Remove piece from the board
        old_idx_row, old_idx_col = piece.current_position
        new_state.board.pop(piece.current_position_notation)
        current_player.remove_piece(piece)

        # Make new piece
        new_piece = piece.copy()
        new_piece.current_position = next_position
        _, new_idx_col = next_position
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

        new_piece.moves += 1

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
            new_state.moves_without_progress = 0

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

        if move in {"O-O", "O-O-O"}:
            king_moves = current_player.king.generate_possible_moves(state)

            if move not in king_moves:
                return commons.Result(error="Cannot castle")

            king_idx_row, king_idx_col = current_player.king.current_position

            if move == "O-O":
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
                increase_moves_without_progress=False,
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

        if state.moves_without_progress >= 50:
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
            possible_moves = self.generate_possible_moves(self.current_state)
            print("Choose one of the following possible moves:")

            for idx, possible_move in enumerate(possible_moves):
                print(f"{idx + 1}. {possible_move[0]}")

            move = utils.prompt_input("Enter your move: ")
            result = self.evaluate_move(self.current_state, move)

            if not result.success and not result.data:
                print(result.error)
                continue

            self._current_state = t.cast(ChessState, result.data)
            break
