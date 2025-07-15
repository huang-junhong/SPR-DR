import json


def load_json(path: str) -> dict:
    """
    Loads and parses a JSON file from the specified path.

    Inputs:
    --------
        path (str): The file path to the JSON file.

    Returns:
    --------
        dict: The contents of the JSON file as a dictionary.
              Returns an empty dictionary if the file cannot be read or parsed.
    """
    
    try:
        with open(path, 'r', encoding='utf-8') as file:
            # Load and parse the JSON file
            data = json.load(file)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        print(f"Error: File not found at path: {path}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file at path: {path}. Error details: {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading the JSON file: {e}")
    
    # Return an empty dictionary if any error occurs
    return {}
