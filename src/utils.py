HISTORY_FILE_PATH = "history.txt"


def clear_history_file() -> None:
    with open(HISTORY_FILE_PATH, "w"):
        pass


def prompt_input(message: str, save_history: bool = True) -> str:
    input_ = input(message).strip()

    if save_history:
        with open(HISTORY_FILE_PATH, "a") as file:
            file.writelines(input_ + "\n")

    return input_


def make_hash_number(value: str) -> int:
    # hash_ = hashlib.sha1(value.encode()).hexdigest()
    # return int(hash_, 16)  # Debug hash consistency. Overflows with mypyc
    return hash(value)
