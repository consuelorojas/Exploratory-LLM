import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'hello_world'))
from main import main


import sys
from io import StringIO
import pytest
from your_module import main  # Replace 'your_module' with the actual module name where the main function is defined


def test_main_function(capsys):
    """Test that running the main function prints 'hello world' to standard output."""
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"
