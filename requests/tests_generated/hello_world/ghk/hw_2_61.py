import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code/hello_world')


from main import main


import io
from unittest.mock import patch
import pytest

def main() -> None:
    print("hello world")

@pytest.mark.parametrize(
    "expected_output",
    ["hello world"],
)
@patch('sys.stdout', new_callable=io.StringIO)
def test_main(mock_stdout, expected_output):
    """Test that the main function prints 'hello world' to stdout."""
    main()
    assert mock_stdout.getvalue().strip() == expected_output
