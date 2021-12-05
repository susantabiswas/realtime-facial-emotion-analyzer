# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Tests for media utils."""
# ===================================================

from emotion_analyzer.exceptions import InvalidImage
import pytest
from emotion_analyzer.media_utils import (
    convert_to_dlib_rectangle,
    convert_to_rgb,
    load_image_path,
)
import numpy as np
import cv2
import dlib


def test_convert_to_dlib_rectangle():
    """ Check if dlib rectangle is created properly"""
    bbox = [1, 2, 3, 4]
    dlib_box = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
    assert convert_to_dlib_rectangle(bbox) == dlib_box


def test_load_image_path():
    """ Check if exception is thrown when an invalid array is given"""
    path = "data/sample/1.jpg"
    img = cv2.imread(path)
    img = convert_to_rgb(img)
    loaded_img = load_image_path(path)
    assert np.all(loaded_img == img) == True


def test_convert_to_rgb_exception():
    """ Check if exception is thrown when an invalid array is given"""
    # create a dummy image
    img = np.zeros((100, 100, 5))
    with pytest.raises(InvalidImage):
        convert_to_rgb(img)


def test_convert_to_rgb(img1_data):
    """ Check if RGB conversion happens correctly"""
    rgb = cv2.cvtColor(img1_data, cv2.COLOR_BGR2RGB)
    converted_img = convert_to_rgb(img1_data)
    assert np.all(rgb == converted_img) == True
