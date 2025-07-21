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


# Alternative way using pytest's monkeypatch fixture
@pytest.fixture
def mock_stdout(monkeypatch):
    buffer = StringIO()
    monkeypatch.setattr(sys, 'stdout', buffer)
    return buffer


def test_main_function_with_mock(mock_stdout):
    """Test that the main function prints 'hello world' to standard output."""
    main()
    assert mock_stdout.getvalue().strip() == "hello world"
