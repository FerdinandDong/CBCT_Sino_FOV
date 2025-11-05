import os

def print_directory_tree(path, indent=''):
    """
    Prints the directory tree starting from the given path,
    only showing .py and .yaml files, and excluding .git directory.
    """
    if '.git' in path:  # Skip .git directory
        return

    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            if item.endswith(('.py', '.yaml')):
                print(indent + '├── ' + item)
        elif os.path.isdir(item_path):
            print(indent + '├── ' + item)  # Show directory names
            print_directory_tree(item_path, indent + '│   ')

if __name__ == "__main__":
    current_directory = os.getcwd()
    print_directory_tree(current_directory)