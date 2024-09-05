import time
from src import engines
from src import games
from src import utils


class Copilot:
    def select_game(self) -> None:
        game_options = [game_type.value for game_type in games.GameType]

        if len(game_options) == 1:
            print(f'Selecting "{game_options[0]}"')
            self.game = games.make_game(game_options[0])
            return

        while True:
            try:
                game = utils.prompt_input(
                    f"Enter the game ({', '.join(game_options)}): "
                )
                self.game = games.make_game(game)
                break
            except ValueError as exc:
                print(str(exc))

    def select_engine(self) -> None:
        engine_options = [engine_type.value for engine_type in engines.EngineType]

        if len(engine_options) == 1:
            print(f'Selecting "{engine_options[0]}"')
            self.engine = engines.make_engine(engine_options[0], self.game)
            return

        while True:
            try:
                engine = utils.prompt_input(
                    f"Enter the game engine ({', '.join(engine_options)}): "
                )
                self.engine = engines.make_engine(engine, self.game)
                break
            except ValueError as exc:
                print(str(exc))

    def run(self) -> None:
        utils.clear_history_file()
        self.select_game()
        self.select_engine()

        self.game.setup()
        self.engine.setup()

        while not self.game.is_terminal(self.game.current_state):
            self.game.print_state(self.game.current_state)
            start = time.time()
            self.engine.suggest_move()
            duration = time.time() - start
            print(f"Suggestion calculated in {round(duration, 4)} seconds")
            self.game.make_move()

        print("Game over")
        self.game.print_state(self.game.current_state)


if __name__ == "__main__":
    copilot = Copilot()
    copilot.run()
