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

# Alternatively, you can use pytest's built-in redirecting functionality
@pytest.fixture(autouse=True)
def capture_stdout(monkeypatch):
    """Redirect sys.stdout to a StringIO object for testing purposes."""
    stdout_capture = StringIO()
    monkeypatch.setattr(sys, 'stdout', stdout_capture)
    yield stdout_capture

def test_main_function_with_fixtures(capture_stdout):
    """Test that the main function prints 'hello world' to stdout using fixtures."""
    main()
    assert capture_stdout.getvalue().strip() == "hello world"
