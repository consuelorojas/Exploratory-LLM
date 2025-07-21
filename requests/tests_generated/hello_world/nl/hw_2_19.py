import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'hello_world'))
from main import main


import pytest
from io import StringIO
import sys

def main() -> None:
    print("hello world")

@pytest.fixture
def capsys():
    with StringIO() as captured_output, redirect_stdout(captured_output):
        yield captured_output

@contextmanager
def redirect_stdout(new_target):
    old_targets = sys.stdout
    try:
        sys.stdout = new_target
        yield new_target
    finally:
        sys.stdout = old_targets

def test_main(capsys):
    main()
    captured = capsys.getvalue().strip()
    assert captured == "hello world"
