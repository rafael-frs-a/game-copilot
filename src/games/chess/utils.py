# Dictionaries used to help converting input into move or converting move into suggested input
position_to_notation: dict[tuple[int, int], str] = {}
notation_to_position: dict[str, tuple[int, int]] = {}
columns = ["a", "b", "c", "d", "e", "f", "g", "h"]

for idx_row in range(8):
    for idx_col in range(8):
        notation = f"{columns[idx_col]}{8 - idx_row}"
        position_to_notation[(idx_row, idx_col)] = notation
        notation_to_position[notation] = (idx_row, idx_col)
