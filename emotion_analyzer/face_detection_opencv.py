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


    def model_inference(self, image) -> List:
        # Run the face detection model on the image to get
        # bounding box coordinates
        # The model expects input as a blob, create input image blob
        img_blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), [104, 117, 123], False, False
        )
        # Feed the input blob to NN and get the output layer predictions
        self.face_detector.setInput(img_blob)
        detections = self.face_detector.forward()

        return detections


    def detect_faces(self, image, conf_threshold: float = 0.7) -> List[List[int]]:
        """Performs facial detection on an image. Uses OpenCV DNN based face detector.
        Args:
            image (numpy array):
            conf_threshold (float, optional): Threshold confidence to consider
        Raises:
            InvalidImage: When the image is either None or
            with wrong number of channels.

        Returns:
            List[List[int]]: List of bounding box coordinates
        """
        if not is_valid_img(image):
            raise InvalidImage
        # To prevent modification of orig img
        image = image.copy()
        height, width = image.shape[:2]

        # Do a forward propagation with the blob created from input img
        detections = self.model_inference(image)
        # Bounding box coordinates of faces in image
        bboxes = []
        for idx in range(detections.shape[2]):
            conf = detections[0, 0, idx, 2]
            if conf >= conf_threshold:
                # Scale the bbox coordinates to suit image
                x1 = int(detections[0, 0, idx, 3] * width)
                y1 = int(detections[0, 0, idx, 4] * height)
                x2 = int(detections[0, 0, idx, 5] * width)
                y2 = int(detections[0, 0, idx, 6] * height)

                if self.crop_forehead:
                    y1 = y1 + int(height * self.shrink_ratio)
                # openCv detector can give a lot of false bboxes
                # when the image is a zoomed in face / cropped face
                # This will get rid of atleast few, still there can be other
                # wrong detections present!
                if self.is_valid_bbox([x1, y1, x2, y2], height, width):
                    bboxes.append([x1, y1, x2, y2])

        return bboxes

    def is_valid_bbox(self, bbox: List[int], height: int, width: int) -> bool:
        """Checks if the bounding box exists in the image.

        Args:
            bbox (List[int]): Bounding box coordinates
            height (int):
            width (int):

        Returns:
            bool: Whether the bounding box is valid
        """
        for idx in range(0, len(bbox), 2):
            if bbox[idx] < 0 or bbox[idx] >= width:
                return False
        for idx in range(1, len(bbox), 2):
            if bbox[idx] < 0 or bbox[idx] >= height:
                return False
        return True

    def __repr__(self):
        return "FaceDetectorOPENCV <model_loc=str>"


if __name__ == "__main__":
    ############# Sample Usage #############
    # ob = FaceDetectorOpenCV(model_loc="models", crop_forehead=False)
    # img = cv2.imread("data/media/8.jpg")

    # # import numpy as np
    # # img = np.zeros((100,100,5), dtype='float32')
    # bboxes = ob.detect_faces(convert_to_rgb(img), conf_threshold=0.99)

    # for bbox in bboxes:
    #     cv2.imshow("Test", draw_bounding_box(img, bbox))
    #     cv2.waitKey(0)

    pass