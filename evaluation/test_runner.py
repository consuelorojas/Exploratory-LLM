import subprocess
import json
import tempfile
import os

def run_tests(test_file_path):
    import tempfile
    import os
    import json
    import subprocess

    result = {
        "tests_run": 0,
        "test_pass_rate": 0.0,
        "timeout": False
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmpfile:
        report_path = tmpfile.name

    try:
        # Infer subfolder (e.g., hello_world)
        if "tests_generated" in test_file_path:
            subfolder = test_file_path.split("tests_generated/")[1].split("/")[0]
        else:
            subfolder = "."

        # Path to code/hello_world or similar
        code_dir = os.path.abspath(os.path.join("code", subfolder))

        # Make sure PYTHONPATH includes the code directory
        env = os.environ.copy()
        env["PYTHONPATH"] = code_dir

        # Run pytest with the working dir set to the folder containing the test file
        test_dir = os.path.dirname(test_file_path)

        subprocess.run([
            "pytest",
            os.path.basename(test_file_path),  # just the filename
            "--json-report",
            f"--json-report-file={report_path}",
            "--disable-warnings",
            "-q"
        ],
        cwd=test_dir,  # run inside the test directory
        env=env,
        check=False,
        timeout=120)

        # Load JSON report
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)
                summary = report.get("summary", {})
                total = summary.get("total", 0)
                passed = summary.get("passed", 0)

                result["tests_run"] = total
                result["test_pass_rate"] = round(passed / total, 2) if total > 0 else 0.0

    except subprocess.TimeoutExpired:
        print(f"Timeout: test execution took too long for {test_file_path}")
        result["timeout"] = True

    except Exception as e:
        print(f"Error running tests: {e}")

    finally:
        if os.path.exists(report_path):
            os.remove(report_path)

    return result
