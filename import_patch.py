import os
from pathlib import Path
import re

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

        if project == "digits_classifier":
            path_to_add = "/home/consuelo/Documentos/GitHub/Exploratory-LLM/code"
        elif project == "hello_world":
            path_to_add = "/home/consuelo/Documentos/GitHub/Exploratory-LLM/code/hello_world"
        else:
            print(f"Unknown project: {project} — skipping")
            continue

        import_snippet = (
            "import sys\n"
            f"sys.path.append('{path_to_add}')\n\n"
        )

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Remove any existing sys.path.append or import sys/pathlib lines
        cleaned_lines = [
            line for line in lines
            if not re.search(r"(sys\.path\.append|import\s+sys|import\s+pathlib)", line)
        ]

        # Prepend the correct import snippet
        new_content = import_snippet + "".join(cleaned_lines)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        print(f"Patched: {file_path}")
