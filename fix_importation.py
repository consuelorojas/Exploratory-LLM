from pathlib import Path

BASE_TEST_DIR = Path("requests/tests_generated").resolve()

for file_path in BASE_TEST_DIR.glob("**/*.py"):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Only act if the wrong pattern exists
    if ".parent[4]" in content:
        new_content = content.replace(
            ".parent[4]",
            ".parent.parent.parent"
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        print(f"Fixed: {file_path}")
    else:
        print(f"No fix needed: {file_path}")
