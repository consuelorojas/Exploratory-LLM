import coverage
import importlib.util
import sys
import os

def run_coverage(test_file, code_file):
    cov = coverage.Coverage()
    cov.start()

    try:
        # Import the test file dynamically
        spec = importlib.util.spec_from_file_location("test_module", test_file)
        test_module = importlib.util.module_from_spec(spec)
        sys.modules["test_module"] = test_module
        spec.loader.exec_module(test_module)
    except Exception as e:
        print(f"Error running tests for coverage: {e}")
        return 0.0
    finally:
        cov.stop()
        cov.save()

    try:
        # Report coverage for the code file only
        analysis = cov.analysis2(code_file)
        total = len(analysis[1]) + len(analysis[3])
        covered = len(analysis[1])
        return round(covered / total, 2) if total > 0 else 0.0
    except coverage.CoverageException as ce:
        print(f"Coverage analysis error: {ce}")
        return 0.0
