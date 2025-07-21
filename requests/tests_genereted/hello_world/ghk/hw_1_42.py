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

# Alternatively, you can use pytest's built-in redirect_stdout fixture
@pytest.fixture
def capture_stdout():
    old_out = sys.stdout
    try:
        new_out = StringIO()
        sys.stdout = new_out
        yield new_out
    finally:
        sys.stdout = old_out

def test_main_function_with_capture(capture_stdout):
    """Test that the main function prints 'hello world' to stdout."""
    main()
    assert capture_stdout.getvalue().strip() == "hello world"
