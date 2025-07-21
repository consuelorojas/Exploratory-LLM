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
        yield new_target
    finally:
        sys.stdout = old_targets

def test_main(capsys):
    main()
    assert capsys == "hello world\n"

# Alternative way using pytest's built-in capture functionality
@pytest.mark.parametrize("expected_output", ["hello world"])
def test_hello_world(capfd, expected_output):
    main()
    captured = capfd.readouterr().out.strip()
    assert captured == expected_output

