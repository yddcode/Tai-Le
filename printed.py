# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 10:03:29 2019

@author: HP
"""

# -*- coding: utf-8 -*-

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical       #one-hot 
from keras.callbacks import Callback
from sklearn import metrics
import matplotlib.pyplot as plt
import sys
import time


#-----define choice GPU-ID,using GPU-ID,ex,0,1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#---------------------------------
time_start = time.time()
batch_size = 128
num_classes = 35
epochs = 20

# input image dimensions
img_rows, img_cols = 64, 64

# the data, split between train and test sets
def read_image(img_name):
    im = Image.open(img_name).convert('L')
    data = np.array(im)
    return data

images = []
for fn in os.listdir('train'):
    if fn.endswith('.jpg'):
        fd = os.path.join('train',fn)
        images.append(read_image(fd))
print('load success!')
X_train = np.array(images)
print (X_train.shape)

y_train = np.loadtxt('train label.txt')
print (y_train.shape)

images1 = []
for fn in os.listdir('test'):
    if fn.endswith('.jpg'):
        fd = os.path.join('test',fn)
        images1.append(read_image(fd))
print('load success!')
X_test = np.array(images1)
print (X_test.shape)

y_test = np.loadtxt('test label.txt')
print (y_test.shape)

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#------define model------------------
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=(3, 3)))
# model.add(Conv2D(64, kernel_size=(3, 3)))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(32, kernel_size=(3, 3)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#--------------------------

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])

H = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
          
y_test_pred = model.predict(X_test)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#-----change one-hot to labels

y_test1 = [np.argmax(one_hot)for one_hot in y_test]
y_test_pred1 = [np.argmax(one_hot)for one_hot in y_test_pred]
time_end = time.time()

#--------------------------------

np.savetxt('Y_test.csv', y_test1, delimiter = ',') 
np.savetxt('Y_test_pred.csv', y_test_pred1, delimiter = ',') 

C = confusion_matrix(y_test1, y_test_pred1)
np.savetxt('confusion_matrix.csv', C, delimiter = ',') 

classify_report = metrics.classification_report(y_test1, y_test_pred1,digits=4)
confusion_matrix = metrics.confusion_matrix(y_test1, y_test_pred1)
overall_accuracy = metrics.accuracy_score(y_test1, y_test_pred1)
acc_for_each_class = metrics.precision_score(y_test1, y_test_pred1, average=None)
average_accuracy = np.mean(acc_for_each_class)
score = metrics.accuracy_score(y_test1, y_test_pred1)

N = epochs
plt.figure()
plt.plot(np.arange(0,N),H.history['loss'],label='loss')
plt.plot(np.arange(0,N),H.history['acc'],label='train_acc')
plt.title('Training Loss and Accuracy on yinshuadehong classifier')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig('yuanshicnn.png')
output=sys.stdout
outputfile=open("yuanshicnn.txt","a")
sys.stdout=outputfile
print('classify_report : \n', classify_report)
print('confusion_matrix : \n', confusion_matrix)
print('acc_for_each_class : \n', acc_for_each_class)
print('average_accuracy: {0:f}'.format(average_accuracy))
print('overall_accuracy: {0:f}'.format(overall_accuracy))
print('score: {0:f}'.format(score))
print('totally cost', time_end-time_start)


