import os
from pathlib import Path

BASE_DIR = Path("requests").resolve()

for root, _, files in os.walk(BASE_DIR):
    for file in files:
        if not file.endswith(".py"):
            continue

        file_path = Path(root) / file
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        modified = False
        new_lines = []

        for line in lines:
            # Strip comments and whitespace
            stripped = line.strip().split("#")[0].strip()
            if stripped == "from your_module import main":
                print(f"Removing line in: {file_path}")
                modified = True
                continue
            new_lines.append(line)

        if modified:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

