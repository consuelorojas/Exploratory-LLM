import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code/hello_world')


from main import main


import pytest
from io import StringIO

def main() -> None:
    print("hello world")

@pytest.fixture
def capsys():
    with open('output.txt', 'w') as f:
        old_stdout = sys.stdout
        try:
            sys.stdout = f
            yield
        finally:
            sys.stdout = old_stdout

def test_hello_world(capsys):
    main()
    captured = capsys.readouterr().out.strip()
    assert captured == "hello world"
