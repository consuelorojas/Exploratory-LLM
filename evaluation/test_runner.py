import subprocess
import json
import tempfile
import os

def run_tests(test_file_path):
    result = {
        "tests_run": 0,
        "test_pass_rate": 0.0,
        "timeout": False # default value, no timeout 
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmpfile:
        report_path = tmpfile.name

    try:
        subprocess.run([
            "pytest",
            test_file_path,
            "--json-report",
            f"--json-report-file={report_path}",
            "--disable-warnings",
            "-q"
        ], check=False, timeout=120) # enforce a 2 min limit of computation

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
