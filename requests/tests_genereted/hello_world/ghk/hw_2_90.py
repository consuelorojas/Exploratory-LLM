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
    """Test the output of the main function."""
    # When
    main()

    # Then
    assert mock_stdout.getvalue() == expected_output[0]
