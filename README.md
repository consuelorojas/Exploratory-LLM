# Exploratory-LLM

A toolkit for experimenting with and evaluating Large Language Model (LLM) prompts and workflows.

---

##  Repository Structure
```.
├── batch_process.py # Utilities for batch processing prompts and results
├── run_evaluation.py # Core script to run evaluation workflows
├── requirements.txt # Python dependencies
├── pytest.ini # PyTest configuration
├── .python-version # Python version specifier (pyenv)
├── gherkin-reference.pdf # Reference guide for Gherkin syntax
├── notebooks/ # Jupyter notebooks for interactive experimentation
├── prompt/ # Prompt definitions and templates
├── requests/ # Example requests to LLM APIs
├── results/ # Storage for generated LLM outputs and evaluation data
├── post-processing/ # Scripts or tools for processing results post-evaluation
├── no_patched_results/ # Raw or original results before any patching is applied
├── evaluation/ # Evaluation scripts or datasets
└── .gitignore # Git ignore configuration
```

## 🛠 Usage Overview

### 0. Set Up
To set up the project, first make sure you have Python installed (preferably Python 3.8 or higher). Then, install the required libraries using pip:

```bash
python pip install -r requirements.txt
```

### 1. Generate Unit Tests
Inside `requests/`, run `to_unit_test.py` to generate unit tests from your prompts.  
This script sends requests to an Ollama instance running on **port 8001** of your server.

Example:
```bash
python requests/to_unit_test.py
```
Generated tests will be saved in the designated results directory.

### 2. Post-Processing
Before running the tests, use the scripts in post-processing/ to fix any missing or incorrect dependencies across the generated unit test files.
For example:
```bash
python post-processing/fix_importations.py
```

### 3. Evaluate Results
Use batch_process.py to run all the unit tests in parallel.
This step automatically calls run_evaluation.py internally to compute metrics and aggregate results.
```bash
python batch_process.py
```


