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
    with StringIO() as captured_output, \
         redirect_stdout(captured_output):
        yield captured_output.getvalue()

def test_main(capsys):
    main()
    assert "hello world\n" == sys.stdout.read()

class RedirectStdout:
    def __init__(self):
        self._stdout = None
        self._string_io = StringIO()

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._string_io

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout

def test_main_with_context_manager():
    with RedirectStdout() as redirect:
        main()
        assert "hello world\n" == redirect._string_io.getvalue()

from unittest.mock import patch
import builtins

@patch('builtins.print')
def test_hello_world(mock_print):
    main()
    mock_print.assert_called_once_with("hello world")
