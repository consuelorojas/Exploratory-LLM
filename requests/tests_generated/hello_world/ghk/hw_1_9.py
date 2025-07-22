import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code/hello_world')


from main import main


from io import StringIO
import pytest
from your_module import main  # Replace 'your_module' with the actual name of your module

def test_main_function(capsys):
    """Test that the main function prints 'hello world' to stdout."""
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"

# Alternatively, you can use pytest's built-in capture functionality
@pytest.fixture(autouse=True)
def redirect_stdout(monkeypatch):
    """Redirect sys.stdout for testing purposes."""
    monkeypatch.setattr(sys, 'stdout', StringIO())

def test_main_function_alt():
    """Test that the main function prints 'hello world' to stdout (alternative)."""
    main()
    assert sys.stdout.getvalue().strip() == "hello world"
