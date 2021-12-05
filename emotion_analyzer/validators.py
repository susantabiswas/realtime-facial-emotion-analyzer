# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Helper methods for validation."""
# ===================================================
import os


def is_valid_img(image) -> bool:
    """Checks if an image is valid or not.

    Args:
        image (numpy array): [description]

    Returns:
        bool: [description]
    """
    return image is None or not (len(image.shape) != 3 or image.shape[-1] != 3)


def path_exists(path: str = None) -> bool:
    """Checks if a path exists.

    Args:
        path (str, optional): [description]. Defaults to None.

    Returns:
        bool: [description]
    """
    if path and os.path.exists(path):
        return True
    return False
