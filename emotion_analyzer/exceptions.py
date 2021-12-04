# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Custom Exceptions"""
# ===================================================


class ModelFileMissing(Exception):
    """Exception raised when model related file is missing.

    Attributes:
        message: (str) Exception message
    """

    def __init__(self):
        self.message = "Model file missing!!"


class NoFaceDetected(Exception):
    """Raised when no face is detected in an image

    Attributes:
        message: (str) Exception message
    """

    def __init__(self) -> None:
        self.message = "No face found in image!!"


class MultipleFacesDetected(Exception):
    """Raised when multiple faces are detected in an image

    Attributes:
        message: (str) Exception message
    """

    def __init__(self) -> None:
        self.message = "Multiple faces found in image!!"


class InvalidImage(Exception):
    """Raised when an invalid image is encountered based on array dimension
    Attributes:
        message: (str) Exception message
    """

    def __init__(self) -> None:
        self.message = "Invalid Image!!"


class PathNotFound(Exception):
    """Raised when the path doesn't exist
    Attributes:
        message: (str) Exception message
    """

    def __init__(self) -> None:
        self.message = "Path couldn't be found. Please check!!"


class FaceMissing(Exception):
    """Raised when face is not found in an image
    Attributes:
        message: (str) Exception message
    """

    def __init__(self) -> None:
        self.message = "Face not found!!"
