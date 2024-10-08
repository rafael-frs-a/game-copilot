import os
import torch
from torch.optim import Adam  # type: ignore[attr-defined]
from functools import cached_property
from src.engines.alphazero import constants
from src.engines.alphazero.model import GameNet


class Optimizer(Adam):
    def __init__(self, model: GameNet) -> None:
        self.model = model
        super().__init__(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.load_state()

    @cached_property
    def file_path(self) -> str:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(
            current_dir,
            constants.PROGRESS_FOLDER,
            self.model.game.__class__.__name__,
            "optimizer.pth",
        )

    def load_state(self) -> None:
        if os.path.exists(self.file_path):
            self.load_state_dict(
                torch.load(
                    self.file_path, weights_only=True, map_location=constants.DEVICE
                )
            )

    def save_state(self) -> None:
        directory = os.path.dirname(self.file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        torch.save(self.state_dict(), self.file_path)
