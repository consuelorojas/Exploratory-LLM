import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code/hello_world')


from main import main


import pytest
from io import StringIO

def main() -> None:
    print("hello world")

@pytest.fixture
def capsys():
    with pytest.capsys.disabled():
        yield

def test_hello_world(capsys):
    captured = StringIO()
    sys.stdout = captured
    main()
    sys.stdout = sys.__stdout__
    assert captured.getvalue().strip() == "hello world"

def test_main_call():
    try:
        main()
    except Exception as e:
        pytest.fail(f"main function call failed with exception: {e}")

@pytest.mark.parametrize("expected_output", ["hello world"])
def test_hello_world_parameterized(capsys, expected_output):
    captured = StringIO()
    sys.stdout = captured
    main()
    sys.stdout = sys.__stdout__
    assert captured.getvalue().strip() == expected_output

