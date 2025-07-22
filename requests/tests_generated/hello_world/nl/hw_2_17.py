import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code/hello_world')


from main import main


import pytest
from io import StringIO

def main() -> None:
    print("hello world")

@pytest.fixture
def capsys():
    with StringIO() as captured_output, redirect_stdout(captured_output):
        yield captured_output.getvalue()

def test_main(capsys):
    main()
    assert "hello world\n" == capsys

class RedirectStdout:
    def __init__(self):
        self._stdout = sys.stdout
        self._string_io = StringIO()

    def __enter__(self):
        sys.stdout = self._string_io
        return self._string_io

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout

def test_main_with_redirect_stdout():
    with RedirectStdout() as f:
        main()
        assert "hello world\n" == f.getvalue()

# Using pytest's built-in capsys fixture
@pytest.mark.parametrize("expected_output", ["hello world"])
def test_hello_world(capsys, expected_output):
    main()
    captured = capsys.readouterr().out.strip()
    assert captured == expected_output

