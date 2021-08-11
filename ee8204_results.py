# -*- coding: utf-8 -*-
"""
EE8204 Metrics

Created on Mon Jul 26 14:47:31 2021

@author: Daniel
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd

plt.close('all')

root_path = r'C:\Users\Daniel\Documents\python_code'

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

#make nice graphic of images from the dataset
fig, ax = plt.subplots(2,2)
all_path_1 = random.sample(all_img_paths,2)
all_img_1 = plt.imread(all_path_1[0])
ax[0,0].imshow(all_img_1)
ax[0,0].set_ylabel('ALL')
all_img_2 = plt.imread(all_path_1[1])
ax[0,1].imshow(all_img_2)
nmrl_path_1 = random.sample(nmrl_img_paths,2)
nmrl_img_1 = plt.imread(nmrl_path_1[0])
ax[1,0].imshow(nmrl_img_1)
ax[1,0].set_ylabel('NMRL')
nmrl_img_2 = plt.imread(nmrl_path_1[1])
ax[1,1].imshow(nmrl_img_2)


##########      Confusion Matrix        ##########
cm = np.load('alexnet_lr1E_4_2021,07,26-10,35_CONF_MAT.npy')
labels = ['ALL', 'NMRL']

'''fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
cax = ax.matshow(cm, cmap='viridis')
plt.title('Testing Confusion Matrix')

for (i,j),z in np.ndenumerate(cm):
    ax.text(j,i, '{:0.1f}'.format(z), ha='center', va='center')

fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()'''

#alexnet

norm_cm = cm / cm.astype(np.float).sum(axis=1)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
cax = ax.matshow(norm_cm)
plt.title('Normalized Testing Confusion Matrix - AlexNet')

for (i,j),z in np.ndenumerate(norm_cm):
    ax.text(j,i, '{:0.3f}'.format(z), ha='center', va='center')

cbar = fig.colorbar(cax)
cbar.remove()
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.show()

#vgg16

vg_cm = np.load('vgg16_lr1E_4_2021,07,27-17,04_CONF_MAT.npy')

num_all = vg_cm[0].sum()
num_nmrl = vg_cm[1].sum()

norm_vg_cm = vg_cm / vg_cm.astype(np.float).sum(axis=0)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
dax = ax.matshow(norm_vg_cm)

for (i,j),z in np.ndenumerate(norm_vg_cm):
    ax.text(j,i, '{:0.3f}'.format(z), ha='center', va='center')

cbar = fig.colorbar(dax)
cbar.remove()
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Normalized Testing Confusion Matrix - VGG16')
plt.show()

#resnet50
rn_cm = np.load('resnet50_lr1E_4_2021,07,28-09,16_CONF_MAT.npy')

norm_rn_cm = rn_cm / rn_cm.astype(np.float).sum(axis=1)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
cax = ax.matshow(norm_rn_cm)

for (i,j),z in np.ndenumerate(norm_rn_cm):
    ax.text(j,i, '{:0.3f}'.format(z), ha='center', va = 'center')
    
cbar = fig.colorbar(cax)
cbar.remove()
ax.set_xticklabels(['']+labels)
ax.set_yticklabels(['']+labels)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Normalized Testing Confusion Matrix - ResNet50')
plt.show()

##########      Loss Curves         ##########
epochs = range(1,26)
loss = np.load('alexnet_lr1E_4_2021,07,26-10,35_LOSS.npy')
val_loss = np.load('alexnet_lr1E_4_2021,07,26-10,35_VAL_LOSS.npy')

vg_loss = np.load('vgg16_lr1E_4_2021,07,27-17,04_LOSS.npy')
vg_val_loss = np.load('vgg16_lr1E_4_2021,07,27-17,04_VAL_LOSS.npy')

rn_loss = np.load('resnet50_lr1E_4_2021,07,28-09,16_LOSS.npy')
rn_val_loss = np.load('resnet50_lr1E_4_2021,07,28-09,16_VAL_LOSS.npy')

plt.figure()
plt.title('Loss')
plt.plot(epochs, loss, 'b-', label='AlexNet')
plt.plot(epochs, val_loss, 'b--')
plt.plot(epochs, vg_loss, 'r-', label='VGG16')
plt.plot(epochs, vg_val_loss, 'r--')
plt.plot(epochs, rn_loss, 'k-', label='ResNet50')
plt.plot(epochs, rn_val_loss, 'k--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

##########      Accuracy Curves         ##########
acc = np.load('alexnet_lr1E_4_2021,07,26-10,35_ACC.npy')
val_acc = np.load('alexnet_lr1E_4_2021,07,26-10,35_VAL_ACC.npy')

vg_acc = np.load('vgg16_lr1E_4_2021,07,27-17,04_ACC.npy')
vg_val_acc = np.load('vgg16_lr1E_4_2021,07,27-17,04_VAL_ACC.npy')

rn_acc = np.load('resnet50_lr1E_4_2021,07,28-09,16_ACC.npy')
rn_val_acc = np.load('resnet50_lr1E_4_2021,07,28-09,16_VAL_ACC.npy')

plt.figure()
plt.title('Accuracy')
plt.plot(epochs, acc, 'b-',label='AlexNet')
plt.plot(epochs, val_acc, 'b--')
plt.plot(epochs, vg_acc, 'r-', label='VGG16')
plt.plot(epochs, vg_val_acc, 'r--')
plt.plot(epochs, rn_acc, 'k-', label='ResNet50')
plt.plot(epochs, rn_val_acc, 'k--')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

##########      Precision Curves        ##########
prec = np.load('alexnet_lr1E_4_2021,07,26-10,35_PRECISION.npy')
val_prec =np.load('alexnet_lr1E_4_2021,07,26-10,35_VAL_PRECISION.npy')

vg_prec = np.load('vgg16_lr1E_4_2021,07,27-17,04_PRECISION.npy')
vg_val_prec = np.load('vgg16_lr1E_4_2021,07,27-17,04_VAL_PRECISION.npy')

rn_prec = np.load('resnet50_lr1E_4_2021,07,28-09,16_PRECISION.npy')
rn_val_prec = np.load('resnet50_lr1E_4_2021,07,28-09,16_VAL_PRECISION.npy')

plt.figure()
plt.title('Precision')
plt.plot(epochs, prec, 'b-',label='AlexNet')
plt.plot(epochs, val_prec, 'b--')
plt.plot(epochs, vg_prec, 'r-', label='VGG16')
plt.plot(epochs, vg_val_prec, 'r--')
plt.plot(epochs, rn_prec, 'k-', label='ResNet50')
plt.plot(epochs, rn_val_prec, 'k--')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.show()
