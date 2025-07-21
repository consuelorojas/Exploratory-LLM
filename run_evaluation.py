import json
import os
import argparse
import sys

from evaluation import compilation, test_runner, coverage

def run_evaluation(code_file, test_file, results_file):
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    results = {}

    # 1. Check Compilation
    results["compilation_success"] = compilation.check_compilation(test_file)

    # 2. Run Tests and Get Pass/Fail Count
    if results["compilation_success"]:
        test_results = test_runner.run_tests(test_file)
        results.update(test_results)

        # 3. coverage percent if not time out.
        if test_results.get("timeout", False):
            results["coverage_percent"] = 0.0
        else:
            results["coverage_percent"] = coverage.run_coverage(test_file, code_file)

    else:
        results["test_pass_rate"] = 0
        results["tests_run"] = 0
        results['timeout'] = False #not time out, just didn't compile
        results["coverage_percent"] = 0.0



    # Save results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation complete. Results saved to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GPT-generated tests against code.")
    parser.add_argument("code_file", type=str, help="Path to the source code file")
    parser.add_argument("test_file", type=str, help="Path to the test file")
    parser.add_argument("results_file", type=str, help="Path to save evaluation results")
    args = parser.parse_args()

    run_evaluation(args.code_file, args.test_file, args.results_file)
