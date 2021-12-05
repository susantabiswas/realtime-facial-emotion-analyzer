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