import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'hello_world'))
from main import main


import sys
from io import StringIO
import pytest
from your_module import main  # Replace 'your_module' with the actual name of your module

def test_main_function(capsys):
    """Test that the main function prints 'hello world' to stdout."""
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"

# Alternatively, you can use pytest's built-in redirecting functionality
@pytest.fixture(autouse=True)
def capture_stdout(monkeypatch):
    """Redirect sys.stdout for testing purposes."""
    buffer = StringIO()
    monkeypatch.setattr(sys, 'stdout', buffer)
    yield buffer

def test_main_function_with_capture(capture_stdout):
    """Test that the main function prints 'hello world' to stdout."""
    main()
    assert capture_stdout.getvalue().strip() == "hello world"
