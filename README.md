# About

This project consists of an action suggestion engine for turn-based games.

It currently implements the following games:
- Tic-tac-toe;
- Chess:
  - It does not implement the threefold repetition rule. However, it implements the 50-move draw rule, which is enough to prevent infinite games.

For action suggestion, it currently implements the following engines:
- Monte Carlo Tree Search (MCTS).

# Stack

The project uses the following technologies:
- [Black](https://pypi.org/project/black/);
- [Mypy](https://pypi.org/project/mypy/).

A compiled version of the project can be generated with `python setup.py build_ext --inplace` for some performance improvement.
