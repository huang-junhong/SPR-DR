import os
from typing import List


def read_txt_lines(path: str, encoding: str = "utf-8", strip_line_break: bool = True) -> List[str]:
    """
    Reads a text file line by line and returns a list of lines.

    Inputs
    --------
        path (str): The file path to the text file.
        encoding (str): The encoding of the file (default: "utf-8").
        strip_line_break (bool): If True, removes line break characters from the end of each line (default: True).

    Returns
    --------
        List[str]: A list of lines from the file.
    """

    # Ensure the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at path: {path}")

    try:
        # Read file content line by line
        with open(path, "r", encoding=encoding) as file:
            content = file.readlines()

        # Optionally strip line break characters
        if strip_line_break:
            content = [line.rstrip('\n') for line in content]

        return content

    except Exception as e:
        # Handle unexpected exceptions and re-raise them for visibility
        raise Exception(f"Error reading file at path: {path}. Details: {e}")


def read_txt_full(path: str, encoding: str = "utf-8") -> str:
    """
    Read txt file all content.
    The escape will keep.

    Inputs
    --------
        path (str): The path for read txt file.
        encoding (str): The txt file encode format, default: utf-8.

    Returns
    --------
        Str: The file content read from path.
    """    


    with open(path, 'r', encoding=encoding) as file:
        content = file.read()
    return content



