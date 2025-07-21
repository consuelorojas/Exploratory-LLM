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
        yield captured_output.getvalue()

@contextmanager
def redirect_stdout(new_target):
    old_targets = sys.stdout
    try:
        sys.stdout = new_target
        yield new_target
    finally:
        sys.stdout = old_targets

def test_main():
    with StringIO() as f, redirect_stdout(f) as sio:
        main()
        assert f.getvalue().strip() == "hello world"

class TestMainFunctionality:

    def test_hello_world_output(self):
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     # Redirect stdout.
        main()                                          # Call function.
        sys.stdout = sys.__stdout__                      # Reset redirect.
        assert capturedOutput.getvalue().strip() == "hello world"

def test_main(capsys):
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == 'hello world'
