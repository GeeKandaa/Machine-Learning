## Example image machine learning model using a CNN: Tensorflow version.

import glob
import numpy as np
import cv2
import random as rnd
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

####################################################################################################################
## INITIALISATION AND VISUALISATION

## Loading data
path = 'data/archive/chest_xray/'

train_norm_dir = path + 'train/NORMAL/'
train_pneu_dir = path + 'train/PNEUMONIA/'

test_norm_dir = path + 'test/NORMAL/'
test_pneu_dir = path + 'test/PNEUMONIA/'

val_norm_dir = path + 'val/NORMAL/'
val_pneu_dir = path + 'val/PNEUMONIA/'

## Using glob acts as a file type filter to ensure type syncronosity
train_norm_cases = glob.glob(train_norm_dir + '*jpeg')
train_pneu_cases = glob.glob(train_pneu_dir + '*jpeg')

test_norm_cases = glob.glob(test_norm_dir + '*jpeg')
test_pneu_cases = glob.glob(test_pneu_dir + '*jpeg')

val_norm_cases = glob.glob(val_norm_dir + '*jpeg')
val_pneu_cases = glob.glob(val_pneu_dir + '*jpeg')

##train_norm_cases = [x.replace('\\', '/') for x in train_norm_cases]
##train_pneu_cases = [x.replace('\\', '/') for x in train_pneu_cases]
##test_norm_cases = [x.replace('\\', '/') for x in test_norm_cases]
##test_pneu_cases = [x.replace('\\', '/') for x in test_pneu_cases]
##val_norm_cases = [x.replace('\\', '/') for x in val_norm_cases]
##val_pneu_cases = [x.replace('\\', '/') for x in val_pneu_cases]

train_list = []
test_list = []
val_list = []

for i in train_norm_cases:
    train_list.append([i, 0])
for i in train_pneu_cases:
    train_list.append([i, 1])

for i in test_norm_cases:
    test_list.append([i, 0])
for i in test_pneu_cases:
    test_list.append([i, 1])

for i in val_norm_cases:
    val_list.append([i, 0])
for i in val_pneu_cases:
    val_list.append([i, 1])

##DEBUG
##print(train_list)
##print(test_list)
##print(val_list)

## shuffle data
rnd.shuffle(train_list)
rnd.shuffle(test_list)
rnd.shuffle(val_list)

#create dataframes
train_df = pd.DataFrame(train_list, columns=['image','label'])
test_df = pd.DataFrame(test_list, columns=['image','label'])
val_df = pd.DataFrame(val_list, columns=['image','label'])


#visualise data distribution
cont = input('Visualise data? (y/n)')
if cont == 'y':
    plt.figure(figsize=(20,5))

    plt.subplot(1,3,1)
    sb.countplot(train_df['label'])
    plt.title('Training Data')
    
    plt.subplot(1,3,2)
    sb.countplot(test_df['label'])
    plt.title('Testing Data')
    
    plt.subplot(1,3,3)
    sb.countplot(val_df['label'])
    plt.title('Validation Data')
    
    plt.show()

#sample images
    plt.figure(figsize=(20,8))
    for i,img_path in enumerate(train_df[train_df['label'] == 1][0:4]['image']):
        plt.subplot(2,4,i+1)
        plt.axis('off')
        img = plt.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title('Opacity')
    
    for i,img_path in enumerate(train_df[train_df['label'] == 0][0:4]['image']):
        plt.subplot(2,4,4+i+1)
        plt.axis('off')
        img = plt.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title('Normal')

    plt.show()

print('Data visualisation complete')
cont = input('Enter to continue')

###########################################################################################################################
## DATA PROCESSING

# Preprocessing
def process_data(img_path):
    ret = cv2.imread(img_path,0)
    ret = cv2.resize(ret, (196,196))
    ret = ret/255.0
    if len(ret.shape) != 2:
        ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
    ret = np.reshape(ret, (196,196,1))
    return ret

def compose_dataset(df):
    data = []
    labels = []

    for img_path, label in df.values:
        data.append(process_data(img_path))
        labels.append(label)

    return np.array(data), np.array(labels)

print('Processing.',end='',flush=True)
X_train, y_train = compose_dataset(train_df)
print('\rProcessing..',end='',flush=True)
X_test, y_test = compose_dataset(test_df)
print('\rProcessing...')
X_val, y_val = compose_dataset(val_df)

print('Training data shape: {}, Labels shape: {}'.format(X_train.shape, y_train.shape))
print('Testing data shape: {}, Labels shape: {}'.format(X_test.shape, y_test.shape))
print('Validation data shape: {}, Labels shape: {}'.format(X_val.shape, y_val.shape))

# Data generation
datagen = ImageDataGenerator(
    rotation_range = 10,
    zoom_range = 0.1,
    width_shift_range = 0.1,
    horizontal_flip = False,
    vertical_flip = False)

datagen.fit(X_train)

# Convert binary to categorical (process time improvement)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#############################################################################################################################
## Modelling

model = Sequential()

model.add(Conv2D(filters=8, kernel_size=(7,7), padding='same', activation='relu', input_shape=(196, 196, 1)))
model.add(Conv2D(filters=8, kernel_size=(7,7), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(lr=0.0001, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

callback = EarlyStopping(monitor='loss', patience=6)
history = model.fit(datagen.flow(X_train,y_train, batch_size=4), validation_data=(X_test, y_test), epochs = 25, verbose = 1, callbacks=[callback], class_weight={0:6.0, 1:0.5})

################################################################################################################################
##Evaluation

plt.figure(figsize = (20,5))

plt.subplot(1,2,1)
sb.lineplot(x = history.epoch, y = history.history['loss'], color='red', label = 'Loss')
sb.lineplot(x = history.epoch, y = history.history['val_loss'], color='orange', label='Validation Loss')
plt.title('Loss on training vs testing')
plt.legend(loc = 'best')

plt.subplot(1,2,2)
sb.lineplot(x = history.epoch, y = history.history['accuracy'], color = 'blue', label = 'Accuracy')
sb.lineplot(x = history.epoch, y = history.history['val_accuracy'], color = 'green', label = 'Validation Accuracy')
plt.title('Accuracy on training vs testing')
plt.legend(loc = 'best')

plt.show()

################################################################################################################################
## Confusion matrix

y_test_hat = model.predict(X_test, batch_size=4)
y_test_hat = np.argmax(y_test_hat, axis=1)
y_test = np.argmax(y_test, axis=1)

conf_m = confusion_matrix(y_test, y_test_hat)
clas_r = classification_report(y_test, y_test_hat)

plt.figure(figsize = (5,3))
sb.set(font_scale = 1.2)
ax = sb.heatmap(conf_m, annot=True, xticklabels = ['H','P'], yticklabels = ['H', 'P'], cbar=False, cmap='Blues', linewidths=1, linecolor='black', fmt='.0f')
plt.yticks(rotation=0)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
ax.xaxis.set_ticks_position('top')
plt.title('Confusion matrix - test data\n(H - healthy/normal, P - pneumonia)')

plt.show()

################################################################################################################################
## Validation & Report

print('Classification report on test data')
print(clas_r)

y_val_hat = model.predict(X_val, batch_size=4)
y_val_hat = np.argmax(y_val_hat, axis=1)
y_val = np.argmax(y_val, axis=1)

plt.figure(figsize=(20,15))
for i,x in enumerate(X_val):
    plt.subplot(4,4,i+1)
    plt.imshow(x.reshape(196, 196), cmap='gray')
    plt.axis('off')
    plt.title('Predicted: {}, Real: {}'.format(y_val_hat[i], y_val[i])) 
