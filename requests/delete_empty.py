import os
from pathlib import Path

def delete_empty_files(folder_path):
    folder = Path(folder_path)
    for file in folder.iterdir():
        if file.is_file() and file.stat().st_size == 0:
            print(f"Deleting empty file: {file}")
            file.unlink()

# Example usage
delete_empty_files("/home/consuelo/Documentos/GitHub/Exploratory-LLM/requests/tests_genereted/ANN/ghk")