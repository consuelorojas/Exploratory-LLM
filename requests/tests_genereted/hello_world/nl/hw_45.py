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

def test_main(capsys):
    main()
    captured = capsys.readouterr().out.strip()
    assert captured == "hello world"

# Alternative way to capture output without writing to file
import pytest
from io import StringIO
import sys

@pytest.fixture
def capsys():
    class Capturing:
        def __enter__(self):
            self.old_stdout = sys.stdout
            self.new_stdout = StringIO()
            sys.stdout = self.new_stdout
            return self

        def __exit__(self, *args):
            sys.stdout = self.old_stdout

    with Capturing() as capturer:
        yield capturer.new_stdout.getvalue()

def test_main(capsys):
    main()
    captured = capsys().strip()
    assert captured == "hello world"
