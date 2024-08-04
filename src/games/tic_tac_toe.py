import typing as t
from functools import cache
from src import utils
from src.games import commons


class TicTacToePlayer(commons.Player):
    def __init__(self, symbol: str, value: int) -> None:
        self.symbol = symbol
        self.value = value
        self.make_hash()

    def make_hash(self) -> None:
        self._hash = utils.make_hash_number(self.symbol)


class TicTacToeState(commons.GameState):
    def __init__(
        self,
        board: list[list[t.Optional[TicTacToePlayer]]],
        totals: list[int],
        players: list[commons.Player],
        current_player_idx: int,
        winner: t.Optional[TicTacToePlayer] = None,
    ) -> None:
        self.board = board
        self.totals = totals
        self.players = players
        self.current_player_idx = current_player_idx
        self.winner = winner
        self.make_hash()

    def make_hash(self) -> None:
        # Just the board is enough to uniquely identify the game state in tic-tac-toe
        state = ""

        for row in self.board:
            row_state: list[str] = []

            for cell in row:
                if cell:
                    row_state.append(cell.symbol)
                else:
                    row_state.append(".")

            state += "".join(row_state)

        self._hash = utils.make_hash_number(state)

    def copy(self) -> commons.GameState:
        return TicTacToeState(
            [row[:] for row in self.board],
            self.totals[:],
            self.players,
            self.current_player_idx,
            t.cast(t.Optional[TicTacToePlayer], self.winner),
        )


class TicTacToe(commons.Game):
    def setup(self) -> None:
        # Initial state of game's board
        # `None` `None` `None`
        # `None` `None` `None`
        # `None` `None` `None`
        # X = 1
        # O = -1
        board: list[list[t.Optional[TicTacToePlayer]]] = [[None] * 3 for _ in range(3)]
        # Each possible "row"'s total in the board
        # 0: sum of first row
        # 1: sum of second row
        # 2: sum of third row
        # 3: sum of first column
        # 4: sum of second column
        # 5: sum of third column
        # 6: sum of main diagonal
        # 7: sum of secondary diagonal
        # First row adding up to 3 means "X" wins
        # First row adding up to -3 means "O" wins
        totals = [0] * 8

        player_x = TicTacToePlayer("X", 1)
        player_o = TicTacToePlayer("O", -1)
        players = [player_x, player_o]
        self._current_state = TicTacToeState(
            board, totals, t.cast(list[commons.Player], players), 1
        )

    @property
    def current_state(self) -> TicTacToeState:
        return self._current_state

    def _update_state(self, state: TicTacToeState, idx_row: int, idx_col: int) -> None:
        current_player = t.cast(TicTacToePlayer, state.current_player)
        state.board[idx_row][idx_col] = current_player
        win_scores = {
            player.value * 3: player
            for player in t.cast(list[TicTacToePlayer], state.players)
        }

        def update_total(idx: int) -> None:
            state.totals[idx] += current_player.value
            state.winner = state.winner or win_scores.get(state.totals[idx])

        if idx_row == 0:
            update_total(0)

        if idx_row == 1:
            update_total(1)

        if idx_row == 2:
            update_total(2)

        if idx_col == 0:
            update_total(3)

        if idx_col == 1:
            update_total(4)

        if idx_col == 2:
            update_total(5)

        if idx_row == idx_col:  # Main diagonal
            update_total(6)

        if 2 - idx_row == idx_col:  # Secondary diagonal
            update_total(7)

    def _idx_to_move(self, idx_row: int, idx_col: int) -> str:
        # Convert given indexes into a value from the following board
        # 1 2 3
        # 4 5 6
        # 7 8 9
        return str((idx_col + 1) + 3 * idx_row)

    @cache
    def evaluate_move(
        self, state: commons.GameState, move: str
    ) -> commons.Result[commons.GameState]:
        try:
            square = int(move)
        except ValueError:
            return commons.Result(error="Invalid number")

        state = t.cast(TicTacToeState, state)

        if not 1 <= square <= 9:
            return commons.Result(
                error="Invalid number. Chosen number should be between 1 and 9"
            )

        idx_row = (square - 1) // 3  # Should return 0, 1, or 2
        idx_col = square - 1 - idx_row * 3  # Should return 0, 1, or 2

        if state.board[idx_row][idx_col]:
            return commons.Result(
                error="Square already filled. Please choose an empty square"
            )

        new_state = t.cast(TicTacToeState, state.copy())
        new_state.current_player_idx = state.get_next_player_idx()
        self._update_state(new_state, idx_row, idx_col)
        new_state.make_hash()
        return commons.Result(new_state)

    @cache
    def generate_possible_moves(
        self, state: commons.GameState
    ) -> list[tuple[str, commons.GameState]]:
        state = t.cast(TicTacToeState, state)
        possible_moves: list[tuple[str, commons.GameState]] = []

        if state.winner:
            return possible_moves

        for move in range(1, 10):
            result = self.evaluate_move(state, str(move))

            if result.success and result.data:
                possible_moves.append((str(move), result.data))

        return possible_moves

    def print_state(self, state: commons.GameState) -> None:
        state = t.cast(TicTacToeState, state)
        print("Current game state:")

        for row in state.board:
            row_text: list[str] = []

            for cell in row:
                if cell is None:
                    row_text.append("_")
                else:
                    row_text.append(cell.symbol)

            print(" ".join(row_text))

        if state.winner:
            print(f'Player "{t.cast(TicTacToePlayer, state.winner).symbol}" wins')
        elif self.is_terminal(state):
            print("Draw")

    def make_move(self) -> None:
        while True:
            print("Choose a number between 1 and 9 matching an available square")
            print("E.g.:")
            print("1 2 3")
            print("4 5 6")
            print("7 8 9")

            move = utils.prompt_input("Enter your move: ")
            result = self.evaluate_move(self.current_state, move)

            if not result.success and not result.data:
                print(result.error)
                continue

            self._current_state = t.cast(TicTacToeState, result.data)
            break
