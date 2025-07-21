import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'hello_world'))
from main import main


import pytest
from io import StringIO
import sys

def main() -> None:
    print("hello world")

@pytest.fixture
def capsys():
    with open('output.txt', 'w') as f:
        old_stdout = sys.stdout
        try:
            sys.stdout = f
            yield
        finally:
            sys.stdout = old_stdout

def test_main(capsys):
    main()
    captured = capsys.readouterr().out.strip()
    assert captured == "hello world"

# Alternative approach using pytest's built-in capture functionality
@pytest.mark.parametrize("expected_output", ["hello world"])
def test_hello_world(capfd, expected_output):
    main()
    out, err = capfd.readouterr()
    assert out.strip() == expected_output

