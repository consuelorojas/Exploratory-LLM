import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code/hello_world')


from main import main


from io import StringIO
import pytest


def test_main_function(capsys):
    """Test that the main function prints 'hello world' to standard output."""
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"


# Alternatively, you can use pytest's monkeypatch fixture
@pytest.mark.parametrize("expected_output", ["hello world"])
def test_main_function_with_monkeypatch(monkeypatch, expected_output):
    """Test that the main function prints 'hello world' to standard output."""
    fake_stdout = StringIO()
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", fake_stdout)
        main()

    assert fake_stdout.getvalue().strip() == expected_output
