import os
import json
import subprocess
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Directories
CODE_DIR = Path("code/ANN")
TEST_DIR = Path("requests/tests_generated/ANN/nl")
RESULTS_DIR = Path("results/ANN/nl")

# Ensure output dir exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Collect valid test files: hw_<number>*.py
test_files = sorted([
    f for f in TEST_DIR.glob("ann_*.py")
    if re.match(r"ann_\d+.*\.py", f.name)
])

# Prepare tasks
tasks = []
for i, test_file in enumerate(test_files):
    match = re.match(r"ann_(\d+)", test_file.stem)
    if not match:
        print(f"Skipping invalid file name: {test_file.name}")
        continue

    task_num = i
    code_path = CODE_DIR / "main.py"  # adjust if needed
    result_path = RESULTS_DIR / f"results_{task_num}.json"

    if result_path.exists():
        print(f"Skipping hw_{task_num}: result already exists.")
        continue

    tasks.append((task_num, code_path, test_file, result_path))

# Run a single task
def run_task(task):
    task_num, code_path, test_path, result_path = task
    try:
        subprocess.run([
            "python3", "run_evaluation.py",
            str(code_path),
            str(test_path),
            str(result_path)
        ], check=False)

        if result_path.exists():
            with open(result_path) as f_result:
                metrics = json.load(f_result)
                metrics["task_num"] = task_num
                return metrics
    except Exception as e:
        print(f"Error processing ann_{task_num}: {e}")
    return None

# Run tasks in parallel
all_results = []
with ProcessPoolExecutor() as executor:
    futures = {executor.submit(run_task, task): task[0] for task in tasks}
    for future in as_completed(futures):
        result = future.result()
        if result:
            all_results.append(result)

# Save combined results
with open(RESULTS_DIR / "all_results.json", "w") as f:
    json.dump(all_results, f, indent=4)

print(f"✅ Processed {len(all_results)} test cases.")
