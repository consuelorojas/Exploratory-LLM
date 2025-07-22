import os
from pathlib import Path

BASE_TEST_DIR = Path("requests/tests_generated").resolve()

for root, _, files in os.walk(BASE_TEST_DIR):
    for file in files:
        if not file.endswith(".py"):
            continue

        file_path = Path(root) / file

        parts = file_path.parts
        try:
            idx = parts.index("tests_generated")
            project = parts[idx + 1]
        except (ValueError, IndexError):
            print(f"Could not determine project for: {file_path}")
            continue

        # Compose the sys.path.append line to add code/<project> to sys.path
        import_path_line = (
            "import sys\n"
            "import pathlib\n"
            f"sys.path.append(str(pathlib.Path(__file__).parents[3] / 'code' / '{project}'))\n\n"
        )

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove any old import from main statements you don't want, e.g.
        # 'from main import ClassifyDigits' or 'from main import main'
        import_statements_to_remove = [
            "from main import ClassifyDigits",
            "from main import main",
        ]
        for stmt in import_statements_to_remove:
            if stmt in content:
                print(f"Removing old import '{stmt}' in {file_path}")
                content = content.replace(stmt, "")

        # Add the sys.path.append snippet if not present
        if import_path_line.strip() not in content:
            print(f"Adding import path patch to {file_path}")
            content = import_path_line + content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            print(f"Import path patch already present in {file_path}")
