from pathlib import Path
import os

BASE_TEST_DIR = Path("requests/tests_generated").resolve()

for root, _, files in os.walk(BASE_TEST_DIR):
    for file in files:
        if not file.endswith(".py"):
            continue

        file_path = Path(root) / file
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        parts = file_path.parts
        try:
            idx = parts.index("tests_generated")
            project = parts[idx + 1]
        except (ValueError, IndexError):
            print(f"Could not determine project for: {file_path}")
            continue

        # Determine the correct import line based on the project
        if project == "ANN":
            extra_line = "from main import ClassifyDigits\n"
        elif project == "hello_world":
            extra_line = "from main import main\n"
        else:
            print(f"Unknown project type: {project} — skipping {file_path}")
            continue

        # Look for the sys.path.append line and patch the extra import
        for i, line in enumerate(lines):
            if line.startswith("sys.path.append("):
                if i + 1 < len(lines) and lines[i + 1] == extra_line:
                    print(f"Already patched: {file_path}")
                    break
                else:
                    print(f"Patching project import in: {file_path}")
                    lines.insert(i + 1, extra_line)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                    break
