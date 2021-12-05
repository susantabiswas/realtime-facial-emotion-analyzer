# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Class for face detection. Uses face detectors
from dlib.

Usage: python -m emotion_analyzer.face_detection_dlib

Ref: http://dlib.net/cnn_face_detector.py.html
"""
# ===================================================
import os
import sys
from typing import List

import cv2
import dlib

from face_recog.exceptions import InvalidImage, ModelFileMissing
from face_recog.face_detector import FaceDetector
from face_recog.logger import LoggerFactory
from face_recog.media_utils import convert_to_rgb
from face_recog.validators import is_valid_img

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


class FaceDetectorDlib(FaceDetector):
    """Class for face detection. Uses face detectors from dlib.
    Raises:
        ModelFileMissing: [description]
        InvalidImage: [description]
    """

    cnn_model_filename = "mmod_human_face_detector.dat"

    def __init__(self, model_loc: str = "models", model_type: str = "hog"):
        """Constructor

        Args:
            model_loc (str, optional): Path where the models are saved.
                Defaults to 'models'.
            model_type (str, optional): Supports HOG and MMOD based detectors.
                Defaults to 'hog'.

        Raises:
            ModelFileMissing: Raised when model file is not found
        """
        try:
            # load the model
            if model_type == "hog":
                self.face_detector = dlib.get_frontal_face_detector()
            else:
                # MMOD model
                cnn_model_path = os.path.join(
                    model_loc, FaceDetectorDlib.cnn_model_filename
                )
                if not os.path.exists(cnn_model_path):
                    raise ModelFileMissing
                self.face_detector = dlib.cnn_face_detection_model_v1(cnn_model_path)
            self.model_type = model_type
            logger.info("dlib: {} face detector loaded...".format(self.model_type))
        except Exception as e:
            raise e