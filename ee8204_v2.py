# -*- coding: utf-8 -*-
"""
Another attempt at code for my Neural Networks project of Luekemia Classification

Created on Fri Jul 16 18:50:37 2021

@author: Daniel
"""

###########     Library Imports     ##########
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow
import sklearn #called later
import random
import datetime
import sys

cc = True

##########      Import Data         ##########
if cc:
    root_path = sys.argv[1]
else:
    root_path = os.getcwd()

data_path = os.path.join(root_path, 'C-NMC_Leukemia')

train_path = os.path.join(data_path, 'training_data')
val_path = os.path.join(data_path, 'validation_data')
test_path = os.path.join(data_path, 'testing_data')

all_folds = [] #directories with images labeled ALL
nmrl_folds = [] #directories with images labeled NMRL
for fold in os.listdir(train_path):
    all_folds.append(os.path.join(train_path, fold, 'all'))
    nmrl_folds.append(os.path.join(train_path, fold, 'hem'))

#make paths of all the images
all_img_paths = []
for fold in range(len(all_folds)):
    for img in os.listdir(all_folds[fold]):
        all_img_paths.append(os.path.join(all_folds[fold], img))

nmrl_img_paths = []
for fold in range(len(nmrl_folds)):
    for img in os.listdir(nmrl_folds[fold]):
        nmrl_img_paths.append(os.path.join(nmrl_folds[fold], img))

all_num = len(all_img_paths)
nmrl_num = len(nmrl_img_paths)
total_img_num = all_num + nmrl_num

print('Number of ALL images: {}\nNumber of NMRL images: {}\nTotal number of images: {}'.format(
    all_num, nmrl_num, total_img_num))

#put all paths in one long list
total_img_paths = []
total_img_paths.extend(all_img_paths)
total_img_paths.extend(nmrl_img_paths)

info_dicti = {'image_paths':total_img_paths,
               'target': ['filler' for i in range(len(total_img_paths))]}

for i in range(len(total_img_paths)):
    if i < all_num:
        info_dicti['target'][i] = 'All' #ALL classification set to 1
    else:
        info_dicti['target'][i] = 'NMRL' #NMRL classification set to 0

#verify that target assignment was correct
info_df = pd.DataFrame(info_dicti)
print(info_df['target'].value_counts())
print(info_df[all_num - 3:all_num + 3])
'''
#make nice graphic of images from the dataset
fig, ax = plt.subplots(2,2)
all_path_1 = random.sample(all_img_paths,2)
all_img_1 = plt.imread(all_path_1[0])
ax[0,0].imshow(all_img_1)
ax[0,0].set_title('ALL')
all_img_2 = plt.imread(all_path_1[1])
ax[0,1].imshow(all_img_2)
ax[0,1].set_title('ALL')
nmrl_path_1 = random.sample(nmrl_img_paths,2)
nmrl_img_1 = plt.imread(nmrl_path_1[0])
ax[1,0].imshow(nmrl_img_1)
ax[1,0].set_title('NMRL')
nmrl_img_2 = plt.imread(nmrl_path_1[1])
ax[1,1].imshow(nmrl_img_2)
ax[1,1].set_title('NMRL')
'''
##########      Train/Test Split        ##########
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(info_df['image_paths'], info_df['target'],
                                                    test_size=0.5, stratify=info_df['target'], shuffle=True)
train_dicti = {'image_paths':list(X_train),
               'target':list(Y_train)}
train_df = pd.DataFrame(train_dicti)

test_dicti = {'image_paths':list(X_test),
              'target':list(Y_test)}
test_df = pd.DataFrame(test_dicti)

##########      Instantiate ImageDataGenerator      ##########
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=25,
    height_shift_range=25,
    rotation_range=180,
    rescale=1./255)         #rescale is 1./255 because images are uint8

train_generator = train_gen.flow_from_dataframe(
    train_df,
    x_col='image_paths',
    y_col='target',
    target_size=(227,227),
    color_mode='rgb',
    class_mode='binary',
    batch_size=48,
    shuffle=True)

#implement the validation images that were given in the kaggle dataset
val_csv = pd.read_csv(os.path.join(val_path, 'C-NMC_test_prelim_phase_data_labels.csv'))

val_csv['image_paths'] = ''
val_csv['target'] = ''
for i in range(len(val_csv['image_paths'])):
    val_csv.at[i, 'image_paths'] = os.path.join(val_path, 'C-NMC_test_prelim_phase_data', val_csv['new_names'][i])
    if val_csv['labels'][i] == 1:
        val_csv.at[i,'target'] = 'ALL'
    elif val_csv['labels'][i] == 0:
        val_csv.at[i,'target'] = 'NMRL'

val_generator = train_gen.flow_from_dataframe(
    val_csv,
    x_col='image_paths',
    y_col='target',
    target_size=(227,227),
    color_mode='rgb',
    class_mode='binary',
    shuffle=True)

test_generator = train_gen.flow_from_dataframe(
    test_df,
    x_col='image_paths',
    y_col='target',
    target_size=(227,227),
    color_mode='rgb',
    class_mode='binary',
    shuffle=False)

##########      Tensorflow Core         ##########
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

#TODO: add relevant callbacks, metrics, and fine tune model by adding neurons, layers, etc.

def alexnet():
    model = tensorflow.keras.models.Sequential()
    model.add(Conv2D(96, (11,11), strides=4, activation='relu', input_shape=(227,227,3)))
    model.add(MaxPool2D((3,3), strides=2))
    model.add(Conv2D(256, (5,5), padding='valid', activation='relu'))
    model.add(MaxPool2D((3,3), strides=2))
    model.add(Conv2D(384, (3,3), padding='valid', activation='relu'))
    model.add(Conv2D(384, (3,3), padding='valid', activation='relu'))
    model.add(Conv2D(256, (3,3), padding='valid', activation='relu'))
    model.add(MaxPool2D((3,3), strides=2))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(tensorflow.keras.layers.BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

NAME = 'resnet50_lr1E_4_{}'.format(datetime.datetime.now().strftime('%Y,%m,%d-%H,%M'))
tboard = tensorflow.keras.callbacks.TensorBoard(log_dir=r'logs/fit/ee8204_project_v2/{}'.format(NAME))
#cback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_false_negatives', min_delta=2)
    
input_tensor = tensorflow.keras.Input(shape=(227,227,3))

model = tensorflow.keras.Sequential()

model.add(tensorflow.keras.applications.ResNet50(weights=None, input_tensor=input_tensor, classes=2, include_top=False))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1E-5),
              metrics=['accuracy',
                       tensorflow.keras.metrics.Precision(),
                       tensorflow.keras.metrics.FalseNegatives(),
                       tensorflow.keras.metrics.FalsePositives()])

history = model.fit(train_generator, 
                    epochs=25, 
                    validation_data=val_generator, 
                    callbacks=[tboard])

##########      Model Evaluation & Confusion Matrix         ##########
print('Testing .....')
model.evaluate(test_generator)

predictions = model.predict(test_generator)

conf_mat = tensorflow.math.confusion_matrix(test_generator.labels, np.round(predictions))
print(conf_mat)


#make nice graphics of appropriate loss curves, confusion matrix
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)
'''
plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()
plt.show()
'''

##########      Save Evaluation Metrics in Numpy Files      ##########
with open(NAME+'_CONF_MAT.npy', 'wb') as f:
    np.save(f, conf_mat)
    
with open(NAME+'_ACC.npy', 'wb') as f:
    np.save(f, history.history['accuracy'])

with open(NAME+'_VAL_ACC.npy', 'wb') as f:
    np.save(f, history.history['val_accuracy'])
    
with open(NAME+'_PRECISION.npy', 'wb') as f:
    np.save(f, history.history['precision'])
    
with open(NAME+'_VAL_PRECISION.npy', 'wb') as f:
    np.save(f, history.history['val_precision'])
    
with open(NAME+'_LOSS.npy', 'wb') as f:
    np.save(f, history.history['loss'])
    
with open(NAME+'_VAL_LOSS.npy', 'wb') as f:
    np.save(f, history.history['val_loss'])