import os
from pathlib import Path
import fnmatch
import re

BASE_TEST_DIR = Path("requests/").resolve()



for root, _, files in os.walk(BASE_TEST_DIR):
    for file in files:
        if not file.endswith(".py"):
            continue

        file_path = Path(root) / file
        print(file_path)
        # Extract project (ANN, hello_world)
        parts = file_path.parts
        try:
            idx = parts.index("tests_generated")
            project = parts[idx + 1]
        except (ValueError, IndexError):
            print(f"Could not determine project for: {file_path}")
            continue

        import_snippet = (
            "import sys\n"
            "import pathlib\n"
            f"sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / '{project}'))\n\n"
        )

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if import_snippet.strip() not in content:
            print(f"Patching import in: {file_path}")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(import_snippet + content)
        else:
            print(f"Import already present in: {file_path}")