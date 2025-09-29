import os

root_dir = os.getcwd()

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.startswith("._"):
            file_path = os.path.join(dirpath, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")