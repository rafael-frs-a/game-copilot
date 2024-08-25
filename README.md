# About

This project consists of an action suggestion engine for turn-based games.

It currently implements the following games:
- Tic-tac-toe;
- Chess:
  - It does not implement the threefold repetition rule. However, it implements the 50-move draw rule, which is enough to prevent infinite games.

For action suggestion, it currently implements the following engines:
- Monte Carlo Tree Search (MCTS).

![image](https://github.com/user-attachments/assets/f0025229-d2df-4b0d-8f1e-05781bb38fad)

Note: although theoretically useful, MCTS is not usable with the current implementation of chess, as it would take several hours to suggest one good move.

To start the copilot run `python main.py`. The provided inputs are stored in a `history.txt` file at the root.

# Stack

The project uses the following technologies:
- [Black](https://pypi.org/project/black/);
- [Mypy](https://pypi.org/project/mypy/).

A compiled version of the project can be generated with `python setup.py build_ext --inplace` for some performance improvement.
