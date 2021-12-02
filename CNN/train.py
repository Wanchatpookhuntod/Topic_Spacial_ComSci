import tensorflow.keras as k
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
import pickle

data = np.load("other/data.npy")
labels = np.load("other/labels.npy")

split = train_test_split(data, labels, test_size=0.2, random_state=64)
trainX, testX, trainY, testY = split

lb = sklearn.preprocessing.LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

model = k.models.Sequential()

model.add(k.layers.Conv2D(16, (3, 3), input_shape=(32, 32, 3), activation='relu'))
model.add(k.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(k.layers.Dropout(0.1))

model.add(k.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(k.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(k.layers.Dropout(0.1))

model.add(k.layers.Flatten())
model.add(k.layers.Dense(512, activation='relu'))
model.add(k.layers.Dropout(0.5))
model.add(k.layers.Dense(len(lb.classes_), activation='softmax'))

INIT_LR = 0.01
EPOCHS = 70
BS = 128

opt = k.optimizers.Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=BS)

model.save("other/model.h5")

with open("other/class.pickle", "wb") as f:
    f.write(pickle.dumps(lb))

predictions = model.predict(x=testX, batch_size=BS)
print(sklearn.metrics.classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))