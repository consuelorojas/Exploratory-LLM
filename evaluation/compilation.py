import subprocess

def check_compilation(test_file_path):
    try:
        subprocess.run(["python", "-m", "py_compile", test_file_path], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False
