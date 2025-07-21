import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'hello_world'))


import sys
from io import StringIO
import pytest
from your_module import main  # Replace 'your_module' with the actual module name where the main function is defined


def test_main_function(capsys):
    """Test that the main function prints 'hello world' to standard output."""
    captured = capsys.readouterr()
    assert captured.out == ''
    
    main()

    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"
```

Alternatively, you can use `contextlib.redirect_stdout` and `StringIO` for testing:

```python
import sys
from io import StringIO
import pytest
from contextlib import redirect_stdout
from your_module import main  # Replace 'your_module' with the actual module name where the main function is defined


def test_main_function():
    """Test that the main function prints 'hello world' to standard output."""
    f = StringIO()
    with redirect_stdout(f):
        main()

    assert f.getvalue().strip() == "hello world"
