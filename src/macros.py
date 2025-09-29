import os

def read_paths(data_path)->list[str]:
    """ Read available directories and return a list of paths """
    if not os.path.exists(data_path):
        return []
    directories = [
        os.path.join(data_path, item)
        for item in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, item))
    ]
    return directories

