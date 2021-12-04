# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Class for face detection. Uses a MTCNN 
based neural network to get the bounding box coordinates 
for a human face.

Usage: python -m emotion_analyzer.face_detection_mtcnn

You can install mtcnn using PIP by typing "pip install mtcnn"
Ref: https://github.com/ipazc/mtcnn
"""
# ===================================================
import sys
from typing import List

import cv2
from mtcnn import MTCNN

from emotion_analyzer.exceptions import InvalidImage
from emotion_analyzer.face_detector import FaceDetector
from emotion_analyzer.logger import LoggerFactory
from emotion_analyzer.media_utils import convert_to_rgb
from emotion_analyzer.validators import is_valid_img

# Load the custom logger
logger = None
try:
    logger_ob = LoggerFactory(logger_name=__name__)
    logger = logger_ob.get_logger()
    logger.info("{} loaded...".format(__name__))
    # set exception hook for uncaught exceptions
    sys.excepthook = logger_ob.uncaught_exception_hook
except Exception as exc:
    raise exc


class FaceDetectorMTCNN(FaceDetector):
    """Class for face detection. Uses a MTCNN
    based neural network to get the bounding box coordinates
    for a human face.
    """

    def __init__(self, crop_forehead: bool = True, shrink_ratio: int = 0.1):
        """Constructor

        Args:
            crop_forehead (bool, optional): Whether to trim the
                forehead in the detected facial ROI. Certain datasets
                like Dlib models are trained on cropped images without forehead.
                It can useful in those scenarios.
                Defaults to True.
            shrink_ratio (float, optional): Amount of height to shrink
                Defaults to 0.1
        """
        try:
            # load the model
            self.face_detector = MTCNN()
            self.crop_forehead = crop_forehead
            self.shrink_ratio = shrink_ratio
            logger.info("MTCNN face detector loaded...")
        except Exception as e:
            raise e
