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
@pytest.fixture
def mock_stdout():
    old_stdout = sys.stdout
    new_stdout = StringIO()
    yield new_stdout
    sys.stdout = old_stdout

def test_main_function_with_mock(mock_stdout):
    """Test that the main function prints 'hello world' to stdout."""
    sys.stdout = mock_stdout
    main()
    assert mock_stdout.getvalue().strip() == "hello world"
