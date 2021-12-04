# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Helper methods for media operations."""
# ===================================================
from typing import List, Tuple

import cv2
import dlib

from emotion_analyzer.exceptions import InvalidImage
from emotion_analyzer.validators import is_valid_img


def convert_to_rgb(image):
    """Converts an image to RGB format.

    Args:
        image (numpy array): [description]

    Raises:
        InvalidImage: [description]

    Returns:
        [type]: [description]
    """
    if not is_valid_img(image):
        raise InvalidImage
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_to_dlib_rectangle(bbox):
    """Converts a bounding box coordinate list
    to dlib rectangle.

    Args:
        bbox (List[int]): Bounding box coordinates

    Returns:
        dlib.rectangle: Dlib rectangle
    """
    return dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])


def load_image_path(img_path, mode: str = "rgb"):
    """Loads image from disk. Optional mode
    to load in RGB mode

    Args:
        img_path (numpy array): [description]
        mode (str, optional): Whether to load in RGB format.
            Defaults to 'rgb'.

    Raises:
        exc: [description]

    Returns:
        [type]: [description]
    """
    try:
        img = cv2.imread(img_path)
        if mode == "rgb":
            return convert_to_rgb(img)
        return img
    except Exception as exc:
        raise exc


def draw_bounding_box(image, bbox: List[int], color: Tuple = (0, 255, 0)):
    """Used for drawing bounding box on an image

    Args:
        image (numpy array): [description]
        bbox (List[int]): Bounding box coordinates
        color (Tuple, optional): [description]. Defaults to (0,255,0).

    Returns:
        [type]: [description]
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image


def draw_annotation(image, name: str, bbox: List[int], color: Tuple = (0, 255, 0)):
    """Used for drawing bounding box and label on an image

    Args:
        image (numpy array): [description]
        name (str): Label to annotate
        bbox (List[int]): Bounding box coordinates
        color (Tuple, optional): [description]. Defaults to (0,255,0).

    Returns:
        [type]: [description]
    """
    draw_bounding_box(image, bbox, color=color)
    x1, y1, x2, y2 = bbox

    # Draw the label with name below the face
    cv2.rectangle(image, (x1, y2 - 20), (x2, y2), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (x1 + 6, y2 - 6), font, 0.6, (0, 0, 0), 2)


def get_facial_ROI(image, bbox: List[int]):
    """Extracts the facial region in an image
    using the bounding box coordinates.

    Args:
        image ([type]): [description]
        bbox (List[int]): [description]

    Raises:
        InvalidImage: [description]

    Returns:
        [type]: [description]
    """
    if image is None or bbox is None:
        raise InvalidImage if image is None else ValueError
    return image[bbox[1] : bbox[3], bbox[0] : bbox[2], :]


def get_video_writer(video_stream, output_filename: str = "data/output.mp4"):
    """Returns an OpenCV video writer with mp4 codec stream

    Args:
        video_stream (OpenCV video stream obj): Input video stream
        output_filename (str):

    Returns:
        OpenCV VideoWriter:
    """
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        FPS = video_stream.get(cv2.CAP_PROP_FPS)

        # (Width, Height)
        dims = (int(video_stream.get(3)), int(video_stream.get(4)))
        video_writer = cv2.VideoWriter(output_filename, fourcc, FPS, dims)
        return video_writer
    except Exception as exc:
        raise exc
