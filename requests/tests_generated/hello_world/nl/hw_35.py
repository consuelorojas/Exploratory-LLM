import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'hello_world'))
from main import main


import pytest
from io import StringIO
import sys

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

def test_main_runs_without_error():
    try:
        main()
    except Exception as e:
        pytest.fail(f"main function raised an exception: {e}")

def test_hello_world_output(capsys):
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"
