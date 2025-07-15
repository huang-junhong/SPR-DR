import os
import yaml


def write_yaml(content: dict, path: str, encoding: str = "utf-8") -> bool:
    """
    Writes a Python dictionary to a YAML file with the specified encoding.

    Inputs:
    --------
        content (dict): 
            The content to write to the YAML file.
        path (str): 
            The file path (absolute or relative) where the YAML file will be written.
        encoding (str): 
            The encoding to use for writing the file (default: "utf-8").

    Returns:
    --------
        bool: True if the operation is successful, False otherwise.
    """

    # Validate that content is a dictionary
    if not isinstance(content, dict):
        raise ValueError("Content must be a dictionary.")

    try:
        # Extract the directory from the file path
        directory = os.path.dirname(os.path.abspath(path))

        # Ensure the directory exists or create it
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Write the dictionary to the specified YAML file
        with open(path, 'w', encoding=encoding) as file:
            yaml.safe_dump(content, file, default_flow_style=False, allow_unicode=True)

        return True

    except Exception as e:
        print(f"Error writing YAML file at path: {path}. Details: {e}")
        return False

