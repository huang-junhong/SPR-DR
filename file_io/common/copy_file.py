import os
import shutil


def copy_file(ori_file: str, destination: str) -> bool:
    """
    copy ori_file to destination.
    If destination not exist, will create destination.

    Inputs
    --------
        ori_file (str): the file/folder aim to copy.
        destination (str): the destination folder for save copied files.

    Returns
    --------
        success (bool): the flag if copy success.
        
    """
    
    if not os.path.exists(ori_file):
        print(f"")
        return False
    
    if not os.path.exists(destination):
        os.makedirs(destination)

    try:
        if os.path.isfile(ori_file):
            shutil.copy2(ori_file, destination)
        elif os.path.isdir(ori_file):
            shutil.copytree(ori_file, destination, dirs_exist_ok=True)
        return True
    except Exception as e:
        print(f'Error when copy {ori_file} to {destination}.')
        return False