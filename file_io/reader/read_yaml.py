import yaml


def load_yaml(path: str) -> dict:
    """
    Loads and parses a YAML file from the specified path.

    Args:
    --------
        path (str): The file path to the YAML file.

    Returns:
    --------
        dict: The contents of the YAML file as a dictionary.
              Returns an empty dictionary if the file cannot be read or parsed.
    """

    try:
        with open(path, 'r', encoding='utf-8') as file:
            # Load and parse the YAML file
            data = yaml.safe_load(file)
            return data if data is not None else {}
    except FileNotFoundError:
        print(f"Error: File not found at path: {path}")
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file at path: {path}. Error details: {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading the YAML file: {e}")
    
    # Return an empty dictionary if any error occurs
    return {}
