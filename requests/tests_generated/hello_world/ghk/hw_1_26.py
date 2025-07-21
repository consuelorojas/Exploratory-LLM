import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'hello_world'))
from main import main


import sys
from io import StringIO
import pytest
from your_module import main  # Replace 'your_module' with the actual module name where the main function is defined


def test_main_function(capsys):
    """Test that the main function prints 'hello world' to standard output."""
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"


# Alternatively, you can use pytest's built-in redirecting of stdout
@pytest.fixture(autouse=True)
def capture_stdout(monkeypatch):
    """Redirect sys.stdout to a StringIO object for testing purposes."""
    buffer = StringIO()
    monkeypatch.setattr(sys, 'stdout', buffer)
    yield buffer


def test_main_function_with_capture(capture_stdout):
    """Test that the main function prints 'hello world' to standard output using pytest's stdout redirection."""
    main()
    assert capture_stdout.getvalue().strip() == "hello world"
