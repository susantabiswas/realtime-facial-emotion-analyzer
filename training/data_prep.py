import cv2
import numpy as np

def load_images(start_idx, end_idx, base_path):
    # training images
    images = []
    for name in range(start_idx, end_idx):
        img = cv2.imread(base_path + str(name) + '.jpg', 0)
        if img is not None:
            images.append(img)

    return images

# read images from folder
def load_images_folder():
    # training images
    images_train = load_images(0, 28710, 'output/Training/')
    # validation images
    images_cv = load_images(28710, 32299, 'output/PublicTest/')
    # test images
    images_test = load_images(32299, 35888, 'output/PrivateTest/')
    
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
