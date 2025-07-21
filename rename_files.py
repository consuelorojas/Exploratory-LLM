from pathlib import Path

BASE_DIR = Path("requests/tests_generated/hello_world/nl")
all_files = sorted(BASE_DIR.glob("*/*/*.py"))  # e.g., hello_world/nl/test_x.py



for i, path in enumerate(all_files, 1):
    new_name = f"test_{i:04d}.py"
    new_path = path.parent / new_name

    if path.name == new_name:
        print(f"Already named correctly: {path}")
        continue

    print(f"Renaming: {path} -> {new_path}")
    path.rename(new_path)
