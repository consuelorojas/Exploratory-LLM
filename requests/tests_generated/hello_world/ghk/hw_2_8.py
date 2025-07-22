import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code/hello_world')


from main import main


import io
from unittest.mock import patch
import pytest

def test_main_function(capsys):
    """Test that 'hello world' is printed to stdout when running the main function."""
    from your_module import main  # Replace with actual module name
    
    main()
    
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"

# Alternatively, you can use pytest's monkeypatch fixture
@pytest.mark.parametrize("expected_output", ["hello world"])
def test_main_function_with_monkeypatch(monkeypatch, expected_output):
    """Test that 'hello world' is printed to stdout when running the main function."""
    from your_module import main  # Replace with actual module name
    
    captured = io.StringIO()
    
    monkeypatch.setattr(sys, "stdout", captured)
    
    main()
    
    assert captured.getvalue().strip() == expected_output

# Using pytest's capfd fixture
def test_main_function_with_capfd(capfd):
    """Test that 'hello world' is printed to stdout when running the main function."""
    from your_module import main  # Replace with actual module name
    
    main()
    
    captured = capfd.readouterr().out.strip()
    assert captured == "hello world"
