# About

AlphaZero, published in late 2017 by researchers of DeepMind, is the first AI model to surpass human proficiency in multiple games while being trained entirely through self-play simulations, without any game-specific implementation in its model. The published [paper](https://arxiv.org/pdf/1712.01815) presents its training results for three major games: *chess*, *shogi*, and *go*.

# How it works

The reason Monte Carlo Tree Search (MCTS) is slow at finding a good next move given a board state is that it simulates full games in each iteration. This is necessary because reaching a terminal state is the only way to determine the game's outcome and use it to evaluate preceding states. In games with many possible legal states, like *chess*, previous searches are often less useful because the current state may not have been seen before. But what if we could evaluate how good a given board position is for the current player more quickly? That would mean full game simulations during the MCTS search are no longer needed, significantly speeding it up.

AlphaZero combines a modified MCTS with a deep neural network to achieve this. Its neural network functions like a complex image classifier, taking the board state as input and generating two outputs:

1. Policy: An array with the probability of each possible legal move in the game, where higher values suggest the best moves for the given state;
2. Value: A scalar between -1 and 1, evaluating the current board position for the player. A value of 1 indicates a certain win, and -1 indicates a certain loss.

During each MCTS search, AlphaZero uses its deep neural network to evaluate a given state and select the most promising move. The full game simulations are moved from MCTS to training the model. In training, the agent plays against itself many times, starting with mostly random moves. The games and their outcomes are used as data to train the model, adjusting its weights and improving its performance. This process repeats, generating better data each time, creating a virtuous cycle.

# Optimizations

This implementation supports [CUDA](https://developer.nvidia.com/cuda-toolkit) if available, which speeds up forwarding data through the model. It also implements the following:

1. Searching multiple game states at once in MCTS instead of one at a time. This takes advantage of the model's ability to process multiple inputs efficiently, also reducing the overhead of transferring data between CPU and GPU;
2. Using multiprocessing during the self-play phase, allowing more data to be generated for training in the same time period by using more CPU cores.

# How to run it

To use AlphaZero as a move suggestion engine, run `python main.py` and select `alphazero` as the engine. If trained data is available in the game's progress folder, it will load it into the model. Otherwise, it will prompt for the number of hidden layers and residual blocks.

To train AlphaZero, run `python train_alphazero.py`. Similarly, if trained data exists in the game's progress folder, it will load it so training can resume from where it left off. Otherwise, it will prompt for hyperparameter values.

![alphazero-tic-tac-toe](https://github.com/user-attachments/assets/c88e02d6-e25e-42f2-94e8-8917820a6a66)

# Limitations

## Proprietary original implementation

The original AlphaZero implementation by DeepMind is proprietary, and the paper doesn’t disclose many details. This implementation is, at best, inspired by other projects based on what’s outlined in the paper.

## Hardware

It took me around 30 minutes to train 10 chess games in one learning iteration on my machine, which has a 12th Gen Core i7, 16 GB of RAM, and an NVIDIA RTX 3060 GPU. Training more games pushed the memory to 100%, causing disk swapping and significantly slowing it down. Some games finished in 10 minutes, while others took longer. AlphaZero's paper mentions it took 4 hours, or 300k learning iterations, to surpass human-level chess proficiency, and with a larger model. Even if simulating more games in parallel didn’t increase training time, at my current rate, it would take about 17.12 years to achieve similar results.

One potential way to reduce the number of required learning iterations some orders of magnitude would be training the model on public records of chess games, sorted by the Elo rating of the players. This would simulate the self-play games AlphaZero uses to generate training data, where early games lack the best moves, but later ones improve. However, this approach wouldn’t allow AlphaZero to develop its own strategies for most of the training.

## Built-in games

Although the AlphaZero model and MCTS don’t have any game-specific implementation, it still requires a perfect implementation of the game to list legal moves from a given state and run simulations. This limitation was addressed in a later version of the research: MuZero.

# Notes

## Tic-tac-toe

Tic-tac-toe is a good game to test the components of AlphaZero due to its manageable number of legal states. Running MCTS with a randomly initialized model can still produce "somewhat smart moves", and the same applies when running a trained model with zero MCTS iterations.

## Chess policy representation

AlphaZero's paper represents the chess policy as an 8 x 8 x 73 array, with a total of 4,672 possible moves. I chose instead to pre-generate all possible chess moves, including promotions and castling, which results in 4,082 moves.

## The `temperature` hyperparameter

Although AlphaZero's [paper](https://arxiv.org/pdf/1712.01815) doesn't mention a `temperature` hyperparameter, it’s often used to encourage exploration during self-play games, similar to the `temperature` in simulated annealing, except that it doesn't decrease over time. However, the same effect can be achieved by increasing the PUCT constant in the MCTS search.

# Credits

This implementation of AlphaZero was based on the following sources:

- [AlphaZero from Scratch](https://www.youtube.com/watch?v=wuSQpLinRB4): A YouTube video from *freeCodeCamp* where Robert Förster explains step-by-step how to implement AlphaZero for *tic-tac-toe* and *Connect 4*;
- [AlphaZero_Chess](https://github.com/geochri/AlphaZero_Chess): A GitHub repository featuring an AlphaZero implementation for *chess*.
