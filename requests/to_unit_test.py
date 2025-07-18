import pandas as pd
import json
import sys
import os



import req_utils as req

def main():
    ## depending of the type of request, the json to read

    ### from nl to unit test
    df = pd.read_json("pgpt.json")
    test_path = 'test/nl_test'

    ## from specifics gherkins to unit test
    # df = pd.read_json('prompts/ghk1_prompts/all_prompts.json')
    # test_path = 'test/ghk1_test'

    ## from nl summary to unit test
    # df = pd.read_json("simplified_prompts/all_summaries_clean.json")
    # test_path = 'test/summ_test'

    ## from gherkins sum to unit test
    # df = pd.read_json('prompts/ghk2_prompts/all_prompts.json')
    # test_path = 'test/ghk2_test'

    
    for i, row in df.iterrows():
        codefile = f"task_{row["task_num"]}.py"
        prompt = row["prompt"]

        req.llm_request(codefile, test_path, prompt)


if __name__ == "__main__":
    main()