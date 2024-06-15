import os


def get_root_path():
    current_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_path)
