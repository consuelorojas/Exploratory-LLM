import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'hello_world'))


import sys
from io import StringIO
import pytest
from your_module import main  # Replace 'your_module' with the actual module name where the main function is defined


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
