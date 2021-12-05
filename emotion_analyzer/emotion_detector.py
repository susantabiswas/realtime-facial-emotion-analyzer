# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Class for emotion analysis

Usage: python -m emotion_analyzer.emotion_detector

"""
# ===================================================
import numpy as np
from emotion_analyzer.emotion_detector_base import EmotionDetectorBase
from emotion_analyzer.exceptions import InvalidImage, ModelFileMissing, NoFaceDetected
from emotion_analyzer.logger import LoggerFactory
from emotion_analyzer.face_detection_dlib import FaceDetectorDlib
from emotion_analyzer.face_detection_mtcnn import FaceDetectorMTCNN
from emotion_analyzer.face_detection_opencv import FaceDetectorOpenCV
import sys
import os
import dlib
from emotion_analyzer.media_utils import get_facial_ROI
from emotion_analyzer.model_utils import define_model, load_model_weights
from emotion_analyzer.validators import is_valid_img, path_exists

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


class EmotionDetector(EmotionDetectorBase):
    model_weights_filename = "weights.h5"
    keypoints_model_path = "shape_predictor_5_face_landmarks.dat"

    def __init__(self, 
        model_loc: str='models',
        face_detection_threshold: int = 0.99,
        face_detector: str = "dlib",) -> None:

        # construct the model weights path
        model_weights_path = os.path.join(model_loc, EmotionDetector.model_weights_filename)
        # model path for facial keypoint detector, needed for dlib face detection        
        keypoints_model_path = os.path.join(
            model_loc, EmotionDetector.keypoints_model_path
        )
        
        if not (
            path_exists(keypoints_model_path) or path_exists(model_weights_path)
        ):
            raise ModelFileMissing

        # load emotion model
        self.model = define_model()
        self.model = load_model_weights(model_weights_path)

        # select and load face detection model
        if face_detector == "opencv":
            self.face_detector = FaceDetectorOpenCV(
                model_loc=model_loc, crop_forehead=True, shrink_ratio=0.2
            )
        elif face_detector == "mtcnn":
            self.face_detector = FaceDetectorMTCNN(crop_forehead=True, shrink_ratio=0.2)
        else:
            self.face_detector = FaceDetectorDlib()
        self.face_detection_threshold = face_detection_threshold

        self.keypoints_detector = dlib.shape_predictor(keypoints_model_path)


    def detect_emotion(self, img):
        """Detects emotion from faces in an image

            Img -> detect faces -> for each face: detect emotion
        Args:
            img (numpy matrix): input image

        Raises:
            InvalidImage: 
            NoFaceDetected: 

        Returns:
            str: emotion label
        """
        # Check if image is valid
        if not is_valid_img(img):
            raise InvalidImage
            
        image = img.copy()

        emotions = []        
        try:
            bboxes = self.face_detector.detect_faces(image=image)
            if bboxes is None or len(bboxes) == 0:
                raise NoFaceDetected
            
            for bbox in bboxes:
                # extract the current face from image and run emotion detection
                face = get_facial_ROI(image, bbox)
                emotion = self.detect_facial_emotion(face)
                facial_data = { "bbox": bbox, "emotion": emotion}
                emotions.append(facial_data)
        except Exception as excep:
            raise excep

        return emotions


    def detect_facial_emotion(self, face) -> str:
        if not is_valid_img(face):
            raise InvalidImage

        # list of given emotions
        EMOTIONS = ['Angry', 'Disgusted', 'Fearful',
                    'Happy', 'Sad', 'Surprised', 'Neutral']
        
        # detect emotion
        model_output = self.model.predict(face)
        emotion = EMOTIONS[np.argmax(model_output[0])]

        print(model_output)
        return emotion


if __name__ == "__main__":
    from emotion_analyzer.media_utils import load_image_path

    ob = EmotionDetector(
        model_loc="models",
        face_detector="dlib",
    )
    img1 = load_image_path("data/sample/1.jpg")
    ob.detect_facial_emotion(img1)
