import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'hello_world'))


import pytest
from io import StringIO
import sys

def main() -> None:
    print("hello world")

@pytest.fixture
def capsys():
    with StringIO() as captured_output, redirect_stdout(captured_output):
        yield captured_output.getvalue()

@contextmanager
def redirect_stdout(new_target):
    old_target = sys.stdout
    try:
        sys.stdout = new_target
        yield
    finally:
        sys.stdout = old_target

def test_main(capsys):
    main()
    out, err = capsys.readouterr()
    assert out.strip() == "hello world"
