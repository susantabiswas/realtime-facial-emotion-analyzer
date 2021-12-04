from emotion_analyzer.exceptions import InvalidImage
from emotion_analyzer.face_detection_mtcnn import FaceDetectorMTCNN
import numpy as np
import pytest   

def test_detect_face(img2_data, img2_facebox_mtcnn):
    ob = FaceDetectorMTCNN()
    assert img2_facebox_mtcnn == ob.detect_faces(img2_data)

def test_invalid_image():
    ob = FaceDetectorMTCNN()
    img = np.zeros((100,100,5), dtype='float32')
    with pytest.raises(InvalidImage):
        ob.detect_faces(img)