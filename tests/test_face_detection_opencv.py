from emotion_analyzer.exceptions import ModelFileMissing, InvalidImage
from emotion_analyzer.face_detection_opencv import FaceDetectorOpenCV
import pytest 
import numpy as np

def test_invalid_image():
    model_loc = "./models"
    ob = FaceDetectorOpenCV(model_loc=model_loc)
    img = np.zeros((100,100,5), dtype='float32')

    with pytest.raises(InvalidImage):
        ob.detect_faces(img)


def test_bbox_outside_img():
    model_loc = "./models"
    ob = FaceDetectorOpenCV(model_loc=model_loc)
    
    assert ob.is_valid_bbox([0, 0, 100, 100], 10, 10) == False


def test_correct_model_path():
    """
    Test object init with the correct model path
    """
    ob = None
    model_loc = "./models"
    try:
        ob = FaceDetectorOpenCV(model_loc=model_loc)
    except Exception:
        pass
    finally:
        assert isinstance(ob, FaceDetectorOpenCV)


def test_incorrect_model_path():
    """
    Test object init with the incorrect model path
    """
    inccorrect_model_loc = "./wrong_models"
    with pytest.raises(ModelFileMissing):
        _ = FaceDetectorOpenCV(model_loc=inccorrect_model_loc)


def test_detect_face(img2_data, img2_facebox_opencv):
    model_loc = "./models"
    ob = FaceDetectorOpenCV(model_loc=model_loc)
    assert img2_facebox_opencv == ob.detect_faces(img2_data)
