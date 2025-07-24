import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code/hello_world')


from main import main


from io import StringIO
import pytest

def test_main_function(capsys):
    """Test that the main function prints 'hello world' to stdout."""
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"

# Alternatively, you can use pytest's built-in capture functionality
@pytest.fixture
def captured_stdout(monkeypatch):
    buffer = StringIO()
    monkeypatch.setattr(sys, 'stdout', buffer)
    return buffer

def test_main_function_alternative(captured_stdout):
    """Test that the main function prints 'hello world' to stdout."""
    main()
    assert captured_stdout.getvalue().strip() == "hello world"
