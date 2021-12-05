# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Helper methods for media operations."""
# ===================================================
import decimal
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


def draw_bounding_box_annotation(image, label: str, bbox: List[int], color: Tuple = (0, 255, 0)):
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
    cv2.putText(image, label, (x1 + 6, y2 - 6), font, 0.6, (0, 0, 0), 2)


def annotate_warning(warning_text: str, img):
    """Draws warning text at the bottom of screen

    Args:
        warning_text (str): warning label
        img (numpy array): input image
    """


def annotate_emotion_stats(emotion_data, img):
    """Draws a bar chart of emotion labels on top of image

    Args:
        emotion_data (Dict): Emotions and their respective prediction confidence
        img (numpy array): input image
    """
    for index, emotion in enumerate(emotion_data.keys()):
        
        # for drawing progress bar
        cv2.rectangle(img, (100, index * 20 + 10), (100 +int(emotion_data[emotion]), (index + 1) * 20 + 4),
                        (255, 0, 0), -1)
        # for putting emotion labels
        cv2.putText(img, emotion, (10, index * 20 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (7, 109, 16), 2)
        
        emotion_confidence = str(emotion_data[emotion]) + "%"
        # for putting percentage confidence
        cv2.putText(img, emotion_confidence, (105 + int(emotion_data[emotion]), index * 20 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


def draw_emoji(emoji, img):
    """Puts an emoji img on top of another image.

    Args:
        emoji (numpy array): emoji picture
        img (numpy array): input image
    """
    # overlay emoji on the frame for all the channels
    for c in range(0, 3):
        # for doing overlay we need to assign weights to both foreground and background
        foreground = emoji[:, :, c] * (emoji[:, :, 3] / 255.0)
        background = img[350:470, 10:130, c] * (1.0 - emoji[:, :, 3] / 255.0)
        img[350:470, 10:130, c] = foreground + background


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
