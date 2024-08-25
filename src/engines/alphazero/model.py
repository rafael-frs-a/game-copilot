import torch
from torch import nn
from torch.nn import functional as F
from src.engines.alphazero import utils as alphazero_utils
from src.engines.alphazero.game import AlphaZeroGame


class GameNet(nn.Module):
    def __init__(
        self, game: AlphaZeroGame, num_res_blocks: int = 10, num_hidden: int = 256
    ) -> None:
        super().__init__()
        self.game = game
        self.num_res_blocks = num_res_blocks
        self.num_hidden = num_hidden
        self.device = alphazero_utils.get_device()
        board_height, board_width, num_board_planes = self.game.input_tensor_dimensions
        all_game_moves = self.game.generate_all_possible_moves()

        # Initial convolutional block to process the game input
        self.start_block = nn.Sequential(
            nn.Conv2d(
                num_board_planes,
                self.num_hidden,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(self.num_hidden),
            nn.ReLU(),
        )

        # Stack of residual blocks
        self.back_bone = nn.ModuleList(
            [ResBlock(self.num_hidden) for _ in range(self.num_res_blocks)]
        )

        # Policy head for action prediction
        self.policy_head = nn.Sequential(
            nn.Conv2d(self.num_hidden, self.num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_hidden),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                self.num_hidden * board_width * board_height,
                len(all_game_moves),
            ),
            nn.LogSoftmax(
                dim=1
            ),  # Use log-softmax for more stable training with cross-entropy loss
        )

        # Value head for state evaluation
        self.value_head = nn.Sequential(
            nn.Conv2d(self.num_hidden, self.num_hidden // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_hidden // 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.num_hidden // 2 * board_width * board_height, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # Output a scalar value between -1 and 1
        )

        # Move the model to the specified device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        Args:
            x: Input tensor representing the game state.
        Returns:
            policy: Tensor representing action probabilities (logits).
            value: Tensor representing the evaluation of the current state.
        """
        # Pass the input through the initial convolutional block
        x = self.start_block(x)

        # Pass through the stack of residual blocks
        for res_block in self.back_bone:
            x = res_block(x)

        # Get policy and value predictions
        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden: int) -> None:
        """
        Residual block consisting of two convolutional layers with batch normalization and ReLU activation.
        Args:
            num_hidden: Number of hidden channels in the residual block.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        Args:
            x: Input tensor.
        Returns:
            Output tensor after passing through the residual block with skip connection.
        """
        residual = x  # Save the input tensor for the skip connection
        x = F.relu(self.bn1(self.conv1(x)))  # First convolution and activation
        x = self.bn2(self.conv2(x))  # Second convolution
        x += residual  # Add the residual (skip connection)
        return F.relu(x)  # Apply activation after addition
