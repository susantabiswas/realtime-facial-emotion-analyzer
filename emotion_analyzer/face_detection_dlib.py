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

from emotion_analyzer.exceptions import InvalidImage, ModelFileMissing
from emotion_analyzer.face_detector import FaceDetector
from emotion_analyzer.logger import LoggerFactory
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


    def detect_faces(self, image, num_upscaling: int = 1) -> List[List[int]]:
        """Performs facial detection on an image. Works best with
        RGB image. Uses a dlib based detector either HOG or MMOD.

        Args:
            image (numpy array):
            num_upscaling (int, optional): Number of times to upscale
            while detecting faces. Defaults to 1.

        Raises:
            InvalidImage: When the image is either None or
            with wrong number of channels.

        Returns:
            List[List[int]]: List of bounding box coordinates
        """
        if not is_valid_img(image):
            raise InvalidImage
        return [
            self.dlib_rectangle_to_list(bbox)
            for bbox in self.face_detector(image, num_upscaling)
        ]

    def dlib_rectangle_to_list(self, dlib_bbox) -> List[int]:
        """Converts a dlib rectangle / mmod rectangle to
        List(top left x, top left y, bottom right x, bottom right y)

        Args:
            dlib_bbox (dlib.rectangle):

        Returns:
            List[int]: Bounding box coordinates
        """
        # if it is MMOD type rectangle
        if type(dlib_bbox) == dlib.mmod_rectangle:
            dlib_bbox = dlib_bbox.rect
        # Top left corner
        x1, y1 = dlib_bbox.tl_corner().x, dlib_bbox.tl_corner().y
        width, height = dlib_bbox.width(), dlib_bbox.height()
        # Bottom right point
        x2, y2 = x1 + width, y1 + height

        return [x1, y1, x2, y2]

    def __repr__(self):
        return "FaceDetectorDlib"


if __name__ == "__main__":
    
    # Sample Usage
    # ob = FaceDetectorDlib(model_type="hog")
    # img = cv2.imread("data/sample/2.jpg")
    # print(img.shape)
    # bbox = ob.detect_faces(convert_to_rgb(img))
    # print(bbox)
    
    # draw_bounding_box(img, bbox)
    pass
