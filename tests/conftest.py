from emotion_analyzer.media_utils import load_image_path
import pytest


@pytest.fixture
def img1_data():
    img_loc = "data/sample/1.jpg"
    return load_image_path(img_path=img_loc, mode="rgb")


@pytest.fixture
def img2_data():
    img_loc = "data/sample/2.jpg"
    return load_image_path(img_path=img_loc, mode="rgb")


@pytest.fixture
def img2_keypoints():
    keypoints = [[139, 92], [125, 93], [91, 89], [105, 91], [114, 122]]

    return keypoints


@pytest.fixture
def img2_facebox_dlib_hog():
    return [[66, 66, 156, 157]]


@pytest.fixture
def img2_facebox_dlib_mmod():
    return [[70, 61, 152, 144]]


@pytest.fixture
def img2_facebox_opencv():
    # With dlib style fore head cropping
    return [[74, 72, 153, 160]]


@pytest.fixture
def img2_facebox_mtcnn():
    # With dlib style fore head cropping
    return [[76, 60, 151, 155]]

