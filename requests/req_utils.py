import requests
import json
import os

URL_REQ = "http://localhost:8001/v1/completions"
URL_CONTEXT = "http://localhost:8001/v1/ingest/file"
URL_DEL = "http://localhost:8001/v1/ingest"


# add context to the llm
def add_context(file_path, ollamaHost):
    files = {"file": open(f"code/{file_path}", "r")}
    
    response = requests.post(url=ollamaHost, files=files)
    context_id = response.json()["data"][0]["doc_id"]

    return context_id

# add instructions to get the prompt in gherkin syntaxis
def change_to_gherkins_prompt(text):
    add_on = "change the following instructions to Gherkin syntaxis\n"
    return add_on + text


# request prompt for the change of syntaxis.
def request_prompt(prompt, url):
    request = {
        "include_sources": False,
        "prompt": prompt,
        "stream": False,
        "system_prompt": "Answer Only",
        "use_context": True,
    }
    
    response =  requests.post(url=url, json=request)
    if response.status_code == 200:
        data = response.json()
        code = data["choices"][0]["message"]["content"]

    else:
        print("Error:", response.status_code)
        print("Response:", response.text)
        return
    return code

# request code and strip from first part
def request_code(prompt, url):
    
    request = {
        "include_sources": False,
        "prompt": prompt,
        "stream": False,
        "system_prompt": "Return only code",
        "use_context": True,
    }
    
    response =  requests.post(url=url, json=request)
    if response.status_code == 200:
        data = response.json()
        code = data["choices"][0]["message"]["content"]

    else:
        print("Error:", response.status_code)
        print("Response:", response.text)
        return


    code = response.json()["choices"][0]["message"]["content"]
    code = code.strip(r"```python")
    return code


# as its called: delete the context
def delete_context(url_id):
    #response
    requests.delete(url=url_id)

def save_code(code, filename):
    with open(filename, "w") as file:
        file.write(code)


def llm_request(codefile, test_path, prompt):
    test_code_name = f"{test_path}/test_{codefile}"

    if os.path.exists(test_code_name):
        return
    
    if not open(f"code/{codefile}", "r").read(1):
        return
    
    context_id = add_context(codefile, URL_CONTEXT)

    test_code = request_code(prompt, URL_REQ)
    if test_code == None:
        return
    
    save_code(test_code, test_code_name)

    url_id = f"{URL_DEL}/{context_id}"
    delete_context(url_id)
