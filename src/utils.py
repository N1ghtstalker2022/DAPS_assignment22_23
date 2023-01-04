"""Provide useful python functions for other python modules.

Implement functions for creating directories.

"""
import os


def create_dir(directory):
    """Create directory under current path.

    Args:
        directory: String formatted directory name

    """
    cur_path = os.getcwd()
    path = os.path.join(cur_path, directory)
    if os.path.isdir(path):
        return
    else:
        os.mkdir(path)
