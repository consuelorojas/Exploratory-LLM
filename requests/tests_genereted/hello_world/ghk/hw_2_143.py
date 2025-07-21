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

# Alternatively, you can use pytest's built-in capture functionality
@pytest.fixture(autouse=True)
def redirect_stdout():
    old_stdout = sys.stdout
    new_stdout = StringIO()
    try:
        sys.stdout = new_stdout
        yield new_stdout
    finally:
        sys.stdout = old_stdout

def test_main_function_with_redirect(redirect_stdout):
    """Test that the main function prints 'hello world' to stdout."""
    main()
    assert redirect_stdout.getvalue().strip() == "hello world"
