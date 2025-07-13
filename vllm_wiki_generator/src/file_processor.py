import os
from typing import List

def find_code_files(directory: str, excluded_dirs: List[str] = None, code_extensions: List[str] = None) -> List[str]:
    """
    Recursively finds all code files in a directory, excluding specified directories and filtering by file extension.

    Args:
        directory (str): The directory to search.
        excluded_dirs (List[str], optional): A list of directory names to exclude. Defaults to None.
        code_extensions (List[str], optional): A list of code file extensions to include. Defaults to None.

    Returns:
        List[str]: A list of paths to the code files.
    """
    if excluded_dirs is None:
        excluded_dirs = [".git", "node_modules", "__pycache__", ".vscode", "dist", "build"]
    if code_extensions is None:
        code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs", ".jsx", ".tsx", ".html", ".css", ".php", ".swift", ".cs"]

    code_files = []
    for root, dirs, files in os.walk(directory):
        # Exclude specified directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

        for file in files:
            if any(file.endswith(ext) for ext in code_extensions):
                code_files.append(os.path.join(root, file))

    return code_files

def read_file_content(filepath: str) -> str:
    """
    Reads the content of a file.

    Args:
        filepath (str): The path to the file.

    Returns:
        str: The content of the file.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
