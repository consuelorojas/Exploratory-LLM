from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import req_utils as req

NUM_REQUESTS = 20
MAX_WORKERS = 4

def run_single_request(i, prompt, url_request, path_test_files):
    try:
        code = req.request_code(prompt, url_request)
        output_path = os.path.join(path_test_files, f"ann_3_{i}.py")
        req.save_code(code, output_path)
    except Exception as e:
        print(f"[!] Error in request {i}: {e}")

def main():
    # read prompt
    # prompt_path = "/home/consuelo/Documentos/GitHub/Exploratory-LLM/prompt/hello_world/ghk.txt"
    prompt_path = "/home/consuelo/Documentos/GitHub/Exploratory-LLM/prompt/ANN/ghk_rag.txt"


    with open(prompt_path, "r") as f:
        prompt = f.read()
    
    prompt = str(prompt)
    print(prompt)

    # add context
    
    #context_path = "/home/consuelo/Documentos/GitHub/Exploratory-LLM/code/hello_world/main.py"
    context_path = "/home/consuelo/Documentos/GitHub/Exploratory-LLM/code/digits_classifier/main.py"

    # unsilenced for gherkins pdf syntaxis.
    context_path_ghk = "/home/consuelo/Documentos/GitHub/Exploratory-LLM/gherkin-reference.pdf"
    context_id_ghk = req.add_context(context_path_ghk)

    context_id = req.add_context(context_path)

    url_request = "http://localhost:8001/v1/completions"
    path_test_files = "tests_generated/digits_classifier/ghk_rag"
    Path(path_test_files).mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(run_single_request, i, prompt, url_request, path_test_files)
            for i in range(NUM_REQUESTS)
        ]

        # Wrap with tqdm to track progress as futures complete
        for _ in tqdm(as_completed(futures), total=NUM_REQUESTS, desc="Requests completed"):
            pass  # just advance progress bar per completed future

    req.delete_context(context_id)
    req.delete_context(context_id_ghk)

if __name__ == "__main__":
    main()
