import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'hello_world'))


import pytest
from io import StringIO
import sys

def main() -> None:
    print("hello world")

@pytest.fixture
def capsys():
    with StringIO() as captured_output, redirect_stdout(captured_output):
        yield captured_output.getvalue()

@contextmanager
def redirect_stdout(new_target):
    old_targets = sys.stdout
    try:
        sys.stdout = new_target
        yield new_target.stream
    finally:
        sys.stdout = old_targets

class TestMainFunction:

    def test_main_function(self, capsys):
        main()
        captured_output = capsys.read().strip()  # type: ignore
        assert captured_output == "hello world"

def redirect_stdout(new_target):
    old_targets = sys.stdout
    try:
        sys.stdout = new_target
        yield new_target.stream
    finally:
        sys.stdout = old_targets

class TestMainFunction:

    def test_main_function(self, capsys):
        with StringIO() as captured_output, redirect_stdout(captured_output) as _:
            main()
            assert captured_output.getvalue().strip() == "hello world"
