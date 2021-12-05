# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Base class for emotion detectcion

Usage: python -m emotion_analyzer.emotion_detector_base

"""
# ===================================================
from abc import ABC, abstractmethod


class EmotionDetectorBase(ABC):
    @abstractmethod
    def detect_emotion(self, img):
        pass
