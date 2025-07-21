import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'hello_world'))


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
    """Redirect sys.stdout to a StringIO object for testing purposes."""
    stdout_capture = StringIO()
    monkeypatch.setattr(sys, 'stdout', stdout_capture)
    yield stdout_capture

def test_main_function_with_fixture(capture_stdout):
    """Test that the main function prints 'hello world' to stdout using a fixture."""
    main()
    assert capture_stdout.getvalue().strip() == "hello world"
