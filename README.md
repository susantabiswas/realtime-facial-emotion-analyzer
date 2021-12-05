[![HitCount](http://hits.dwyl.io/susantabiswas/facial-emotion-analyzer.svg)](http://hits.dwyl.io/susantabiswas/facial-emotion-analyzer)
# Realtime Emotion Analysis from facial Expressions
Realtime Human Emotion Analysis From facial expressions. It uses a deep Convolutional Neural Network.
The model used achieved an accuracy of 63% on the test data. The realtime analyzer assigns a suitable emoji for the current emotion.

Model implementation was done in keras.<br>

## Some predicted outputs:
<img src ='data/media/1.JPG'  width="430" height="380"><img src ='media/2.JPG'  width="430" height="380"/>

<img src ='data/media/3.JPG'  width="430" height="380"><img src ='media/4.JPG'  width="430" height="380"/>

### Emojis used:
<img src="data/emojis/neutral.png" width="80" height="80">		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="data/emojis/happy.png" width="80" height="80">      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="data/emojis/fearful.png" width="80" height="80">      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="data/emojis/sad.png" width="80" height="80">      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="data/emojis/angry.png" width="80" height="80">      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="data/emojis/surprised.png" width="80" height="80">      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="data/emojis/disgusted.png" width="80" height="80">

### Snapshot from Realtime emotion Analyzer
The model prediction for the given frame was **Neutral** which is evident from the picture.<br>
<img src ='data/media/5.JPG' wdith="640" height="480"/>

## <u>Model Architecture
<img src ='data/media/model_plot.png' >
  
## <u>List of files
`facial Emotions.ipynb` : 
Jupyter notebook with well documented code explaining model preparation from start to training. Can be used for retraining the model.

  
## How to run
Realtime emotion detection, for this run:<br>
```python video_main.py```<br>


## Credits
- Dataset used was from Kaggle fer2013 Challenge [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- Emojis used were from https://emojiisland.com/
