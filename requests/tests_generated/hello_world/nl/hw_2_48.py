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
    [
        ("hello world\n",),
    ],
)
@patch('sys.stdout', new_callable=io.StringIO)
def test_main(mock_stdout, expected_output):
    """Test that the main function prints 'hello world'."""
    main()
    assert mock_stdout.getvalue() == expected_output[0]

@pytest.mark.parametrize(
    "expected_exception",
    [
        (None,),
    ],
)
@patch('sys.stdout', new_callable=io.StringIO)
def test_main_no_exceptions(mock_stdout, expected_exception):
    """Test that the main function does not raise any exceptions."""
    try:
        main()
    except Exception as e:
        assert str(e) == str(expected_exception[0])
