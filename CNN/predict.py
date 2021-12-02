import tensorflow.keras as k
import argparse
import pickle
import cv2
import numpy as np

# argparse = argparse.ArgumentParser()
# argparse.add_argument("-i", "--image")
# args = vars(argparse.parse_args())

# img = cv2.imread(args["image"])

img = cv2.imread("other/tulib_test.jpeg")

img_for_pred = img.copy()
img_for_pred = cv2.resize(img_for_pred, (32, 32))
img_for_pred = k.utils.img_to_array(img_for_pred)
img_for_pred = np.expand_dims(img_for_pred, axis=0)

model = k.models.load_model("other/model.h5")
lb = pickle.loads(open("other/class.pickle", "rb").read())

pred = model.predict(img_for_pred)

i = pred.argmax(axis=1)[0]
lable = lb.classes_[i]

cv2.putText(img, lable, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)
cv2.imshow("out", img)
cv2.waitKey()