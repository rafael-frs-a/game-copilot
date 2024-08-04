import os


def clean() -> None:
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d != "venv"]

        for file in files:
            if file.endswith(".pyd"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed: {file_path}")


clean()
