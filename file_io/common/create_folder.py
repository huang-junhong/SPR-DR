import os


def create_folder(path: str) -> bool:
    """
    Creates a folder at the specified path if it does not already exist.

    Inputs
    --------
        path (str): The directory path to create.

    Returns
    --------
        bool: True if the folder exists or is successfully created, False otherwise.
    """
    
    # Validate that the path is not empty or None
    if not path or not isinstance(path, str):
        raise ValueError("The path must be a non-empty string.")

    try:
        # Use os.makedirs to create the directory, if it doesn't exist
        os.makedirs(path, exist_ok=True)
        return True

    except Exception as e:
        print(f"Error creating folder at path: {path}. Details: {e}")
        return False


