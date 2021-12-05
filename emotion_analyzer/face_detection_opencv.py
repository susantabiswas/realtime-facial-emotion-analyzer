# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Class for face detection. Uses a OpenCV's CNN 
model to get the bounding box coordinates for a human face.

Usage: python -m face_recog.face_detection_opencv
"""
# ===================================================

import os
import sys
from typing import List

import cv2

from emotion_analyzer.exceptions import InvalidImage, ModelFileMissing
from emotion_analyzer.face_detector import FaceDetector
from emotion_analyzer.logger import LoggerFactory
from emotion_analyzer.media_utils import convert_to_rgb, draw_bounding_box
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


class FaceDetectorOpenCV(FaceDetector):
    """Class for face detection. Uses a OpenCV's CNN
    model to get the bounding box coordinates for a human face.

    """

    def __init__(
        self, model_loc="./models", crop_forehead: bool = True, shrink_ratio: int = 0.1
    ):
        """Constructor

        Args:
            model_loc (str, optional): Path where the models are saved.
                Defaults to 'models'.
            crop_forehead (bool, optional): Whether to trim the
                forehead in the detected facial ROI. Certain datasets
                like Dlib models are trained on cropped images without forehead.
                It can useful in those scenarios.
                Defaults to True.
            shrink_ratio (float, optional): Amount of height to shrink
                Defaults to 0.1
        Raises:
            ModelFileMissing: Raised when model file is not found
        """
        # Model file and associated config path
        model_path = os.path.join(model_loc, "opencv_face_detector_uint8.pb")
        config_path = os.path.join(model_loc, "opencv_face_detector.pbtxt")

        self.crop_forehead = crop_forehead
        self.shrink_ratio = shrink_ratio
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            raise ModelFileMissing
        try:
            # load the model
            self.face_detector = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        except Exception as e:
            raise e

