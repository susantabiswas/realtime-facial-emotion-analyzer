<!-- [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fsusantabiswas%2Frealtime-facial-emotion-analyzer&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com) -->

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/susantabiswas/FaceRecog.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/susantabiswas/FaceRecog/context:python)
[![Maintainability](https://api.codeclimate.com/v1/badges/8507a04fe1535a9c224a/maintainability)](https://codeclimate.com/github/susantabiswas/realtime-facial-emotion-analyzer/maintainability)
![Tests](https://github.com/susantabiswas/FaceRecog/workflows/Tests/badge.svg)
[![Build Status](https://app.travis-ci.com/susantabiswas/realtime-facial-emotion-analyzer.svg?branch=master)](https://app.travis-ci.com/susantabiswas/realtime-facial-emotion-analyzer)
[![codecov](https://codecov.io/gh/susantabiswas/realtime-facial-emotion-analyzer/branch/master/graph/badge.svg?token=O7CRXABZEA)](https://codecov.io/gh/susantabiswas/realtime-facial-emotion-analyzer)



# Realtime Emotion Analysis from facial Expressions
Real-time Human Emotion Analysis From facial expressions. It uses a deep Convolutional Neural Network.
The model used achieved an accuracy of 63% on the test data. The realtime analyzer assigns a suitable emoji for the current emotion. 

There are 4 different face detectors for usage. Wrappers for video and webcam processing are provided for convenience.<br><br>

This emotion recognition library is built with ease and customization in mind. There are numerous control parameters to control how you want to use the features, be it face detection on videos, or with a webcam.
<br>

## Table of Contents
- [Sample Output](#sample-output)
- [Architecture](#architecture)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [References](#references)

# Sample Output

## Processed Video
<img src="data/media/output.gif"/><br>

## Processed Images

<img src="data/media/1.jpg" height="320" /><img src="data/media/2.jpg" height="320" />
<img src="data/media/7.jpg" height="320" /><img src="data/media/8.jpg" height="320" />
<img src="data/media/3.jpg" height="320" /><img src="data/media/4.jpg" height="320" />
<img src="data/media/9.jpg" height="320" />
<!-- <img src="data/media/5.jpg" height="320" /><img src="data/media/6.jpg" height="320" /> -->

For emotion recognition, flow is:

    media -> frame -> face detection -> Facial ROI -> Convolutional Neural Network -> Emotion 

These are the major components:
1. **Face Detection**: There are 4 different face detectors with different cropping options.
2. **Emotion Recognition**: Responsible for handling emotion recognition related functionalities from an image.
3. **Utilities**: Methods for handling image, video operations, validations, etc.

<br>

# Setup
There are multiple ways to set this up.
### Clone the repo and install dependencies.<br>
```python
git clone https://github.com/susantabiswas/realtime-facial-emotion-analyzer.git
pip install -r requirements.txt
```

### Docker Image
You can pull the docker image for this project and run the code there.<br>
```docker pull susantabiswas/emotion-analyzer:latest```

### Dockerfile
You can build the docker image from the docker file present in the repo.

```docker build -t <name> .```


# Project Structure
```

realtime-facial-emotion-analyzer/
├── Dockerfile
├── LICENSE
├── README.md
├── data
│   ├── Ubuntu-R.ttf
│   ├── emojis
│   │   ├── angry.png
│   │   ├── disgusted.png
│   │   ├── fearful.png
│   │   ├── happy.png
│   │   ├── neutral.png
│   │   ├── sad.png
│   │   └── surprised.png
│   ├── media
│   │   ├── 1.JPG
│   │   ├── 2.JPG
│   │   ├── 3.JPG
│   │   ├── 4.JPG
│   │   └── model_plot.png
│   └── sample
│       ├── 1.jpg
│       └── 2.jpg
├── emotion_analyzer
│   ├── emotion_detector.py
│   ├── emotion_detector_base.py
│   ├── exceptions.py
│   ├── face_detection_dlib.py
│   ├── face_detection_mtcnn.py
│   ├── face_detection_opencv.py
│   ├── face_detector.py
│   ├── logger.py
│   ├── media_utils.py
│   ├── model_utils.py
│   └── validators.py
├── models
│   ├── mmod_human_face_detector.dat
│   ├── opencv_face_detector.pbtxt
│   ├── opencv_face_detector_uint8.pb
│   └── shape_predictor_5_face_landmarks.dat
├── requirements.txt
├── tests
│   ├── conftest.py
│   ├── test_face_detection_dlib.py
│   ├── test_face_detection_mtcnn.py
│   ├── test_face_detection_opencv.py
│   └── test_media_utils.py
├── training
│   ├── data_prep.py
│   ├── facial Emotions.ipynb
│   └── preprocess.py
└── video_main.py
```

# Usage

### Emotion Recognition
Depending on the use case, whether to aim for accuracy and stability or speed etc., you can pick the face detector. Also, there are customization options inside face detectors to decide the facial ROI.


To analyze facial emotion using a webcam
```python
# Inside project root
import video_main

# You can pick a face detector depending on Acc/speed requirements
emotion_recognizer = EmotionAnalysisVideo(
                        face_detector="dlib",
                        model_loc="models",
                        face_detection_threshold=0.0,
                    )
emotion_recognizer.emotion_analysis_video(
    video_path=None,
    detection_interval=1,
    save_output=False,
    preview=True,
    output_path="data/output.mp4",
    resize_scale=0.5,
)
```

To analyze facial emotion using a video file
```python
# Inside project root
import video_main

# You can pick a face detector depending on Acc/speed requirements
emotion_recognizer = EmotionAnalysisVideo(
                        face_detector="dlib",
                        model_loc="models",
                        face_detection_threshold=0.0,
                    )
emotion_recognizer.emotion_analysis_video(
    video_path='data/sample/test.mp4,
    detection_interval=1,
    save_output=False,
    preview=True,
    output_path="data/output.mp4",
    resize_scale=0.5,
)
```

To register a face using a loaded image 
```python
# Inside project root
from face_recog.media_utils import load_image_path
from face_recog.face_recognition import FaceRecognition

face_recognizer = FaceRecognition(
                    model_loc="models",
                    persistent_data_loc="data/facial_data.json",
                    face_detector="dlib",
                )
img = load_image_path("data/sample/1.jpg")
# Matches is a list containing information about the matches
# for each of the faces in the image
matches = face_recognizer.register_face(image=img, name=name)
```

Face recognition with a webcam feed
```python
# Inside project root
import video_main

face_recognizer = FaceRecognitionVideo(face_detector='dlib')
face_recognizer.recognize_face_video(video_path=None, \
                                    detection_interval=2, save_output=True, \
                                    preview=True, resize_scale=0.25)
```

Face recognition on a video
```python
# Inside project root
import video_main

face_recognizer = FaceRecognitionVideo(face_detector='dlib')
face_recognizer.recognize_face_video(video_path='data/trimmed.mp4', \
                                    detection_interval=2, save_output=True, \
                                    preview=True, resize_scale=0.25)
```

Face recognition using an image
```python
# Inside project root
from face_recog.media_utils import load_image_path
from face_recog.face_recognition import FaceRecognition

face_recognizer = FaceRecognition(
                    model_loc="models",
                    persistent_data_loc="data/facial_data.json",
                    face_detector="dlib",
                )
img = load_image_path("data/sample/1.jpg")
# Matches is a list containing information about the matches
# for each of the faces in the image
matches = face_recognizer.recognize_faces(
                image=img, threshold=0.6
            )
```


There are 4 face detectors namely dlib (HOG, MMOD), MTCNN, OpenCV (CNN). 
All the face detectors are based on a common abstract class and have a common detection interface **detect_faces(image)**.

```python
# import the face detector you want, it follows absolute imports
from face_recog.media_utils import load_image_path
from face_recog.face_detection_dlib import FaceDetectorDlib

face_detector = FaceDetectorDlib(model_type="hog")
# Load the image in RGB format
image = load_image_path("data/sample/1.jpg")
# Returns a list of bounding box coordinates
bboxes = face_detector.detect_faces(image)
```


# Architecture
![architecture](data/media/model_plot.png)<br>
<br>

# References
The awesome work Davis E. King has done: 
http://dlib.net/cnn_face_detector.py.html, 
https://github.com/davisking/dlib-models<br>
You can find more about MTCNN from here: https://github.com/ipazc/mtcnn
<br>
Dataset used was from Kaggle fer2013 Challenge [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
<br>
Emojis used were from https://emojiisland.com/
<br>
Ubuntu font license: https://ubuntu.com/legal/font-licence
