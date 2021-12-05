# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Base class for emotion analysis

Usage: python -m emotion_analyzer.emotion_analysis_base

"""
# ===================================================
from abc import ABC, abstractmethod


class EmotionAnalysisBase(ABC):
    @abstractmethod
    def detect_emotion(self):
        pass
