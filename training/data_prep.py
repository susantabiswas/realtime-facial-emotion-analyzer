import cv2
import numpy as np

# read images from folder
def load_images_folder():
    # training images
    images_train = []
    for name in range(28710):
        img = cv2.imread('output/Training/' + str(name) + '.jpg', 0)
        if img is not None:
            images_train.append(img)

    # validation images
    images_cv = []
    for name in range(28710, 32299):
        img = cv2.imread('output/PublicTest/' + str(name) + '.jpg', 0)
        if img is not None:
            images_cv.append(img)

    # test images
    images_test = []
    for name in range(32299, 35888):
        img = cv2.imread('output/PrivateTest/' + str(name) + '.jpg', 0)
        if img is not None:
            images_test.append(img)

    return images_train, images_cv, images_test
    

# load the images
images_train, images_cv, images_test = load_images_folder()

# change to numpy matrix
images_train = np.array(images_train)
images_cv = np.array(images_cv)
images_test = np.array(images_test)

# save the numpy matrix
np.save('dataset/train_raw.npy', images_train)
np.save('dataset/cv_raw.npy', images_cv)
np.save('dataset/test_raw.npy', images_test)
