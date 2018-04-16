# Realtime Emotion Analysis from facial Expressions
Realtime Human Emotion Analysis From facial expressions. It uses a deep Convolutional Neural Network.
The model used achieved an accuracy of 63% on the test data. The realtime analyzer assigns a suitable emoji for the current emotion.

Model implementation was done in keras.<br>

## Some predicted outputs:
<img src ='media/1.JPG'  width="430" height="380"><img src ='media/2.JPG'  width="430" height="380"/>

<img src ='media/3.JPG'  width="430" height="380"><img src ='media/4.JPG'  width="430" height="380"/>

### Emojis used:
<img src="emojis/neutral.png" width="80" height="80">   <img src="emojis/happy.png" width="80" height="80">   <img src="emojis/fearful.png" width="80" height="80">   <img src="emojis/sad.png" width="80" height="80">   <img src="emojis/angry.png" width="80" height="80">   <img src="emojis/surprised.png" width="80" height="80">   <img src="emojis/disgusted.png" width="80" height="80">

### Snapshot from Realtime emotion Analyzer
The model prediction for the given frame was **Neutral** which is evident from the picture.<br>
<img src ='media/5.JPG' wdith="640" height="480"/>

## <u>Model Architecture
<img src ='media/model_plot.png' >
  
## <u>List of files
`facial Emotions.ipynb` : Jupyter notebook with well documented code explaining model preparation from start to training. Can be used for retraining the model.
<br> `main.py` : main python 
<br> `webcam_utils` :
Code for realtime emotion detection from face
<br> `prediction_utils` :
Code for doing prediction on image saved on disk
<br> `data_prep` :
Code for preparing dataset for training
<br> `preprocess.py` :
Code for saving images from csv file
<br>
  
## How to run
There are two options:
1. Realtime emotion recognition, for this run:<br>
```python main.py emo_realtime```<br>
2. Emotion recognition using image path, for this run:<br>
```python main.py emo_path --path <image path>```
  <br>e.g: ```python main.py emo_path --path saved_images/2.jpg```
  <br><br>If you don't want to specify path then just save the image as **"1.jpg"** inside **saved_images** folder<br> and run:
  ```python main.py emo_path```

## Team Members:
[`Susanta`](https://github.com/susantabiswas/) and [`Sagnik`](https://github.com/SagnikC/)
## Credits
- Dataset used was from Kaggle fer2013 Challenge [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- Emojis used were from https://emojiisland.com/
