import cv2
import sklearn
import numpy as np
import random
import os

path_dataset = "flowers"

list_path_images = []
data = []
labels = []

for label in os.listdir(path_dataset):
    path_label = os.path.join(path_dataset, label)

    for image_file in os.listdir(path_label):
        path_image = os.path.join(path_label, image_file)

        list_path_images.append(path_image)

random.seed(64)
random.shuffle(list_path_images)

for path_image in list_path_images:
    labels.append(path_image.split(os.path.sep)[-2])

    image = cv2.imread(path_image)
    image = cv2.resize(image, (32, 32))
    data.append(image)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

np.save("other/data", data)
np.save("other/labels", labels)

