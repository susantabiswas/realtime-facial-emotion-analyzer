# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Class with methods to do emotion analysis
on video or webcam feed.

Usage: python video_main"""
# ===================================================
import sys
import time
import traceback
from typing import Dict, List

import cv2
import numpy as np

from emotion_analyzer.exceptions import NoNameProvided, PathNotFound
from emotion_analyzer.face_detection_dlib import FaceDetectorDlib
from emotion_analyzer.face_detection_mtcnn import FaceDetectorMTCNN
from emotion_analyzer.face_detection_opencv import FaceDetectorOpenCV
from emotion_analyzer.emotion_detector import EmotionDetector
from emotion_analyzer.logger import LoggerFactory
from emotion_analyzer.media_utils import (
    convert_to_rgb,
    draw_annotation,
    draw_bounding_box,
    get_video_writer,
)
from emotion_analyzer.validators import path_exists

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


class EmotionAnalysisVideo:
    """Class with methods to do emotion analysis on video or webcam feed."""

    def __init__(
        self,
        face_detector: str = "dlib",
        model_loc: str = "models",
        face_detection_threshold: float = 0.8,
    ) -> None:

        self.emotion_analyzer = EmotionDetector(
            model_loc=model_loc,
            face_detection_threshold=face_detection_threshold,
            face_detector=face_detector,
        )
        if face_detector == "opencv":
            self.face_detector = FaceDetectorOpenCV(
                model_loc=model_loc, crop_forehead=True, shrink_ratio=0.2
            )
        elif face_detector == "mtcnn":
            self.face_detector = FaceDetectorMTCNN(crop_forehead=True, shrink_ratio=0.2)
        elif face_detector == "dlib":
            self.face_detector = FaceDetectorDlib()

