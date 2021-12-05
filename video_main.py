# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Class with methods to do emotion analysis
on video or webcam feed.

Usage: python video_main"""
# ===================================================
import sys
import time
import traceback
from typing import Dict, List

import cv2
import numpy as np

from emotion_analyzer.exceptions import PathNotFound
from emotion_analyzer.face_detection_dlib import FaceDetectorDlib
from emotion_analyzer.face_detection_mtcnn import FaceDetectorMTCNN
from emotion_analyzer.face_detection_opencv import FaceDetectorOpenCV
from emotion_analyzer.emotion_detector import EmotionDetector
from emotion_analyzer.logger import LoggerFactory
from emotion_analyzer.media_utils import (
    convert_to_rgb,
    draw_annotation,
    draw_bounding_box,
    get_video_writer,
)
from emotion_analyzer.validators import path_exists

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


class EmotionAnalysisVideo:
    """Class with methods to do emotion analysis on video or webcam feed."""
    emoji_foldername = 'emojis'

    def __init__(
        self,
        face_detector: str = "dlib",
        model_loc: str = "models",
        face_detection_threshold: float = 0.8,
        emoji_loc: str = 'data'
    ) -> None:

        # construct the path to emoji folder
        emoji_path = os.path.join(emoji_loc, EmotionDetector.emoji_foldername)
        # Load the emojis
        self.emojis = self.load_emojis(emoji_path=emoji_path)

        self.emotion_detector = EmotionDetector(
            model_loc=model_loc,
            face_detection_threshold=face_detection_threshold,
            face_detector=face_detector,
        )

    def emotion_analysis_video(
        self,
        video_path: str = None,
        detection_interval: int = 15,
        save_output: bool = False,
        preview: bool = False,
        output_path: str = "data/output.mp4",
        resize_scale: float = 0.5
    ) -> None:

        if video_path is None:
            # If no video source is given, try
            # switching to webcam
            video_path = 0
        elif not path_exists(video_path):
            raise FileNotFoundError

        cap, video_writer = None, None

        try:
            cap = cv2.VideoCapture(video_path)
            # To save the video file, get the opencv video writer
            video_writer = get_video_writer(cap, output_path)
            frame_num = 1
            
            t1 = time.time()
            logger.info("Enter q to exit...")

            emotions = None

            while True:
                status, frame = cap.read()
                if not status:
                    break
                
                try:
                    # Flip webcam feed so that it looks mirrored
                    if video_path == 0:
                        frame = cv2.flip(frame, 2)

                    if frame_num % detection_interval == 0:
                        # Scale down the image to increase model
                        # inference time.
                        smaller_frame = convert_to_rgb(
                            cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                        )
                        # Detect emotion 
                        emotions = self.emotion_detector.detect_emotion(smaller_frame)
                        
                    # Annotate the current frame with emotion detection data
                    self.annotate_emotion_data(emotions, frame, resize_scale)

                    if save_output:
                        video_writer.write(frame)
                    if preview:
                        cv2.imshow("Preview", cv2.resize(frame, (680, 480)))

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                except Exception as exc:
                    raise exc
                frame_num += 1

            t2 = time.time()
            logger.info("Time:{}".format((t2 - t1) / 60))
            logger.info("Total frames: {}".format(frame_num))
            logger.info("Time per frame: {}".format((t2 - t1) / frame_num))

        except Exception as exc:
            raise exc
        finally:
            cv2.destroyAllWindows()
            cap.release()
            video_writer.release()

    
    def annotate_emotion_data(
        self, emotion_data: List[Dict], image, resize_scale: float) -> None:

        for data in emotion_data:
            draw_annotation(image, data['emotion'], int(1 / resize_scale) * np.array(data['bbox']))


    def load_emojis(self, emoji_path:str = 'data//emoji') -> List:
        emojis = []
        logger.info("Emojis loaded...")

        return emojis


if __name__ == "__main__":
    import os
    # SAMPLE USAGE
    from emotion_analyzer.media_utils import load_image_path

    ob = EmotionAnalysisVideo(
            face_detector = "dlib",
            model_loc = "models",
            face_detection_threshold = 0.0,
        )

    img1 = load_image_path("data/sample/1.jpg")
    ob.emotion_analysis_video(
        video_path = None,
        detection_interval = 1,
        save_output = False,
        preview = True,
        output_path = "data/output.mp4",
        resize_scale = 0.5
    )

