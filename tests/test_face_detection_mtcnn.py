import pytest
from emotion_analyzer.exceptions import ModelFileMissing
from emotion_analyzer.face_detection_mtcnn import FaceDetectorMTCNN


def test_detect_face(img2_data, img2_facebox_mtcnn):
    ob = FaceDetectorMTCNN()
    assert img2_facebox_mtcnn == ob.detect_faces(img2_data)
