import hashlib


HISTORY_FILE_PATH = "history.txt"


def clear_history_file() -> None:
    with open(HISTORY_FILE_PATH, "w"):
        pass


def prompt_input(message: str) -> str:
    with open(HISTORY_FILE_PATH, "a") as file:
        input_ = input(message)
        file.writelines(input_ + "\n")
        return input_.strip()


def make_hash_number(value: str) -> int:
    hash_ = hashlib.sha1(value.encode()).hexdigest()
    # return int(hash_, 16)  # Debug hash consistency. Overflows with mypyc
    return hash(hash_)
