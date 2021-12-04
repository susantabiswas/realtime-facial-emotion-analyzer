# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Abstract class for face detectors"""
# ===================================================
from abc import ABC, abstractmethod


class FaceDetector(ABC):
    @abstractmethod
    def detect_faces(self):
        pass
