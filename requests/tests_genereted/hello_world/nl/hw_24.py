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
    with open('output.txt', 'w') as f:
        old_stdout = sys.stdout
        try:
            sys.stdout = f
            yield
        finally:
            sys.stdout = old_stdout

def test_hello_world(capsys):
    main()
    captured = capsys.readouterr().out.strip()
    assert captured == "hello world"

# Alternative way to capture output without writing to a file
@pytest.fixture
def capfd():
    class CapturedOutput:
        def __init__(self, out, err):
            self.out = out.getvalue().strip()

    old_stdout = sys.stdout
    try:
        new_stdout = StringIO()
        sys.stdout = new_stdout
        yield CapturedOutput(new_stdout, None)
    finally:
        sys.stdout = old_stdout

def test_hello_world_alt(capfd):
    main()
    assert capfd.out == "hello world"
