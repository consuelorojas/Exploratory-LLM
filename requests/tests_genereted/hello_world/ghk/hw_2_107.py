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

    # When
    with patch('sys.stdout', new=capturedOutput) as mock_stdout:
        main()

    # Then
    assert capturedOutput.getvalue().strip() == expected_output[0].strip()
