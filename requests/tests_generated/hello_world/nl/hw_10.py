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
