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
@pytest.fixture(autouse=True)
def redirect_stdout(monkeypatch):
    """Redirect sys.stdout for testing purposes."""
    monkeypatch.setattr(sys, 'stdout', StringIO())

def test_main_function_alt():
    """Test that the main function prints 'hello world' to stdout (alternative)."""

    original_stdout = sys.stdout
    captured_output = StringIO()
    try:
        sys.stdout = captured_output
        main()
    finally:
        sys.stdout = original_stdout

    assert captured_output.getvalue().strip() == "hello world"
