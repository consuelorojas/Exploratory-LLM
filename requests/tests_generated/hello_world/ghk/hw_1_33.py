import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code/hello_world')


from main import main


from io import StringIO
import pytest


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
