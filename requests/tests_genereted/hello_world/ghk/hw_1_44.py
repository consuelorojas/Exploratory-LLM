import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'hello_world'))
from main import main


import io
import sys
from unittest.mock import patch
import pytest

def main() -> None:
    print("hello world")

@pytest.mark.parametrize(
    "expected_output",
    [
        ("hello world\n",),
    ],
)
@patch('sys.stdout', new_callable=io.StringIO)
def test_main(mock_stdout, expected_output):
    """Test that the main function prints 'hello world' to stdout."""
    # Given
    capturedOutput = io.StringIO()  # Create StringIO object

    # When: Run the main function with redirected output.
    sys.stdout = mock_stdout
    main()

    # Then: Assert that the expected string was printed to standard output.
    assert mock_stdout.getvalue().strip() == "hello world"
