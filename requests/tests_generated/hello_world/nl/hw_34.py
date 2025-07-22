import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code/hello_world')


from main import main


import pytest
from io import StringIO

def main() -> None:
    print("hello world")

@pytest.fixture
def capsys():
    with StringIO() as captured_output, redirect_stdout(captured_output):
        yield captured_output.getvalue()

@contextmanager
def redirect_stdout(new_target):
    old_targets = sys.stdout
    try:
        sys.stdout = new_target
        yield new_target
    finally:
        sys.stdout = old_targets

def test_main():
    with StringIO() as f, redirect_stdout(f) as sio:
        main()
        assert f.getvalue().strip() == "hello world"

class TestMainFunctionality:

    def test_hello_world_output(self):
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        main()
        sys.stdout = sys.__stdout__
        output = capturedOutput.getvalue().strip()
        assert output == 'hello world'

def test_main(capsys):
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"
