# Realtime Emotion Analysis from facial Expressions
Realtime Human Emotion Analysis From facial expressions. It uses a deep Convolutional Neural Network.
The model used achieved an accuracy of 63% on the test data.

Model implementation was done in keras.<br>
## List of files
#### facial Emotions.ipynb : 
Jupyter notebook with well documented code explaining model preparation from to training. Can be used for retraining the model.
#### main.py :
main python 
#### webcam_utils :
Code for realtime emotion detection from face
#### prediction_utils :
Code for doing prediction on image saved on disk
#### data_prep :
Code for preparing dataset for training
#### preprocess.py :
Code for saving images from csv file

## How to run
There are two options:
1. Realtime emotion detection, for this run
python main.py emo_realtime
2. Emotion detection using image path, for this
python main.py emo_path --path <image path>
e.g: python main.py emo_path --path saved_images/2.jpg
If you don't want to specify path then just save the image as **"1.jpg"** inside **saved_images** folder<br>
  python main.py emo_path

## Credits
- Dataset used was from [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- Emojis used were from https://emojiisland.com/
