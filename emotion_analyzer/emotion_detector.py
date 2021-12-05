# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Class for emotion analysis

Usage: python -m emotion_analyzer.emotion_detector

"""
# ===================================================
from emotion_analyzer.emotion_detector import EmotionDetectorBase
from emotion_analyzer.logger import LoggerFactory
import sys

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
    def __init__(self) -> None:
        super().__init__()

    def detect_emotion(self):
        pass
