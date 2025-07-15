import os

import numpy as np
from typing import List, Union


def get_file_path_by_suffix(folder: str, suffix: Union[str, List[str]]) -> List[str]:
    """
    Recursively finds and returns all file paths in a given folder that match the specified suffix(es).

    Inputs
    --------
        folder (str): The path to the folder where the search will be conducted.
        suffix (Union[str, List[str]]): A single file suffix (e.g., '.txt') or a list of suffixes (e.g., ['.txt', '.csv']).

    Returns
    --------
        List[str]: A list of file paths that match the given suffix(es).
    """
    
    # Ensure suffix is a list for consistency in processing
    if isinstance(suffix, str):
        suffix = [suffix]  
    
    # Normalize suffixes to ensure they all start with a dot (e.g., '.txt')
    suffix = [s if s.startswith('.') else f'.{s}' for s in suffix]

    matched_files = []  # List to store matched file paths

    # Walk through the folder recursively
    for root, _, files in os.walk(folder):
        for file in files:
            # Check if the file ends with any of the specified suffixes
            if any(file.endswith(ext) for ext in suffix):
                matched_files.append(os.path.join(root, file))  # Store the full path

    return sorted(matched_files)

