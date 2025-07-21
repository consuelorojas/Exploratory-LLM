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

@pytest.mark.parametrize("expected", ["hello world"])
def test_main(capsys, expected):
    main()
    captured_output = capsys.read().strip()
    assert captured_output == expected

class TestMainFunctionality:

    def test_hello_world(self, monkeypatch):
        with StringIO() as fake_stdout:
            monkeypatch.setattr(sys, 'stdout', fake_stdout)
            main()
            assert fake_stdout.getvalue().strip() == "hello world"

def test_main_function():
    import io
    capturedOutput = io.StringIO()                  # Create StringIO object
    sys.stdout = capturedOutput                     # Redirect stdout.
    main()                                           # Call function.
    sys.stdout = sys.__stdout__                      # Reset redirect.
    assert capturedOutput.getvalue().strip() == "hello world"
