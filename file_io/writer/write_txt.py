from typing import Union, List


def write_txt(path: str, content: Union[str, List[str]], encoding: str = "utf-8", mode: str = "a+") -> None:
    """
    Writes content to a text file, supporting both string and list of strings as input.

    Inputs
    --------
        path (str): 
            The file path where the content will be written.
        content (Union[str, List[str]]): 
            The content to write. Can be a single string or a list of strings.
        encoding (str): 
            The encoding to use for writing the file (default: "utf-8").
        mode (str): 
            The file mode for writing (default: "a+", append mode).
            Common modes:
            - "w": Overwrite the file.
            - "W+": Overwrite or Create the file.
            - "a": Append to the file.
            - "a+": Append and create the file if it doesn't exist.

    Returns
    --------
        None
    """
    
    # Ensure content is a valid type
    if not isinstance(content, (str, list)):
        raise ValueError("Content must be a string or a list of strings.")

    # Ensure content is iterable if it is a single string
    if isinstance(content, str):
        content = [content]

    try:
        with open(path, mode, encoding=encoding) as file:
            for line in content:
                # Ensure each line ends with a newline character
                if not line.endswith("\n"):
                    line += "\n"
                file.write(line)

    except Exception as e:
        raise Exception(f"Error writing to file at path: {path}. Details: {e}")
