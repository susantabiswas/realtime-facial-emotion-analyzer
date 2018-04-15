import cv2
import sys
from keras.models import load_model
import time
import numpy as np

# loads and resizes an image
def resize_img(image_path):
    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (48, 48))
    #cv2.imwrite(image_path, img)
    return img

# load keras model
our_model = load_model('models/model.h5')
print('model loaded')

save_loc = 'saved_images/1.jpg'

faceCascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_frontalface_default.xml')
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

feelings_faces = []
for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

video_capture = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # mirror the frame
    frame = cv2.flip(frame, 1, 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # required region for the face
        roi_color = frame[y-90:y+h+70, x-50:x+w+50]

        # save the detected face
        cv2.imwrite(save_loc, roi_color)
        #print('reached' + str(f))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # keeps track of waiting time for face recognition
        curr_time = time.time()

        if curr_time - prev_time >=3:
            print('Entered model phase')
            img = cv2.imread(save_loc, 0)
            if img is not None:
                #resize_img(save_loc)
                img = cv2.resize(img, (48, 48))
                print(img.shape)
                #test_img = cv2.imread('save_loc')
                #test_img = np.reshape(test_img, (1, 48,48,1))
                img = np.reshape(img, (1, 48, 48, 1))
                print(img.shape)
                result = our_model.predict(img)
                print(result)
                '''for index, emotion in enumerate(EMOTIONS):
                    cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                    cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),(255, 0, 0), -1)
                    emoji_face = feelings_faces[np.argmax(result[0])]

                for c in range(0, 3):
                    frame[200:320, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + frame[200:320, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)'''
            # save the time when the last face recognition task was done
            prev_time = time.time()
        
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
