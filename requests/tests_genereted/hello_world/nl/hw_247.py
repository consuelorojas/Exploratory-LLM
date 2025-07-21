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
    class Capturing(list):
        def __enter__(self):
            self._stdout = sys.stdout
            sys.stdout = self._stringio = StringIO()
            return self
        
        def __exit__(self, *args):
            self.extend(self._stringio.getvalue().splitlines())
            del self._stringio    
            sys.stdout = self._stdout

    with Capturing() as capturer:
        yield capturer


def test_main(capsys: pytest.fixture) -> None:
    main()
    captured = capsys.readouterr()
    assert "hello world" in captured.out
