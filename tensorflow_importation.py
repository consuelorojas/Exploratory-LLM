import os
from pathlib import Path

TARGET_DIR = Path("requests/tests_generated/digits_classifier").resolve()

for root, _, files in os.walk(TARGET_DIR):
    for file in files:
        if not file.endswith(".py"):
            continue

        file_path = Path(root) / file

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Check if TensorFlow is already imported
        if any("import tensorflow" in line for line in lines):
            print(f"TensorFlow already imported in: {file_path}")
            continue

        # Insert import at the top, after shebang or encoding lines
        insert_index = 0
        for i, line in enumerate(lines):
            if line.startswith("#!") or "coding" in line:
                insert_index = i + 1
            else:
                break

        lines.insert(insert_index, "import tensorflow as tf\n")
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        print(f"Patched TensorFlow import in: {file_path}")
