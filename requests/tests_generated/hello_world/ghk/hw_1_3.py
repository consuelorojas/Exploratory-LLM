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

# Alternatively, you can use pytest's built-in redirect_stdout fixture
@pytest.fixture
def capture_stdout(monkeypatch):
    buffer = StringIO()
    monkeypatch.setattr(sys, 'stdout', buffer)
    return buffer

def test_main_function_with_capture(capture_stdout):
    """Test that the main function prints 'hello world' to stdout."""
    main()
    assert capture_stdout.getvalue().strip() == "hello world"
