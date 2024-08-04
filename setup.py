import glob
from setuptools import setup
from mypyc.build import mypycify

# Run `python setup.py build_ext --inplace` to compile the Python code
files = glob.glob("src/**/*.py", recursive=True)

setup(
    name="game_copilot",
    ext_modules=mypycify(["main.py"] + files),
    zip_safe=False,
)
