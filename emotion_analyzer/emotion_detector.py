# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Class for emotion analysis

Usage: python -m emotion_analyzer.emotion_detector

"""
# ===================================================
import numpy as np
from emotion_analyzer.emotion_detector import EmotionDetectorBase
from emotion_analyzer.exceptions import InvalidImage, ModelFileMissing
from emotion_analyzer.logger import LoggerFactory
import sys
import os
from emotion_analyzer.model_utils import define_model, load_model_weights
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


class EmotionDetector(EmotionDetectorBase):
    model_weights_filename = "weights.h5"

    def __init__(self, model_path='models') -> None:
        # construct the model weights path
        model_weights_path = os.path.join(model_path, EmotionDetector.model_weights_filename)
        if not os.path.exists(model_weights_path):
            raise ModelFileMissing
        
        self.model = define_model()
        self.model = load_model_weights(model_weights_path)


    def detect_emotion(self, img):
        # Check if image is valid
        if not is_valid_img(img):
            raise InvalidImage
            
        image = img.copy()
        
        # list of given emotions
        EMOTIONS = ['Angry', 'Disgusted', 'Fearful',
                    'Happy', 'Sad', 'Surprised', 'Neutral']
        
        # detect emotion
        model_output = self.model.predict(image)
        emotion = EMOTIONS[np.argmax(model_output[0])]

        return emotion
