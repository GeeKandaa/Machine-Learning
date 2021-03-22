## Example image machine learning model using a CNN: Tensorflow version.
from comet_ml import Experiment, ConfusionMatrix
experiment = Experiment(
    api_key="OEqrajWzBdsHvoiWKfOdiAo0c",
    project_name="mixed-ml-experiment-balanced-training-set",
    workspace="geekandaa",
    disabled=True
)
import argparse
import json
import os
import math
from modules import json_comm
import glob
import numpy as np
import cv2
import random as rnd
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

from mpi4py import MPI
comm = MPI.COMM_WORLD

####################################################################################################################
## Load parameters, initialise list to return (for use in taskfarming)

param_vals=json_comm.get_param()
ret = {}

        
####################################################################################################################
## INITIALISATION AND VISUALISATION

## Loading data
path = 'active_data/'

val_healthy_cases = []
val_pneumonia_cases = []
val_covid_cases = []

## Using glob acts as a file type filter to ensure type syncronosity
pneumonia_cases = glob.glob(path + "PN" + '/*.jpeg')
healthy_cases = glob.glob(path + "N" + '/*.jpeg')
covid_cases = glob.glob(path + "CV" + '/*.jpeg')
rnd.shuffle(pneumonia_cases)
rnd.shuffle(healthy_cases)
rnd.shuffle(covid_cases)

pnu_num = len(pneumonia_cases)
nor_num = len(healthy_cases)
cov_num = len(covid_cases)
print(pnu_num)
print(nor_num)
print(cov_num)
chk_num=min(pnu_num,nor_num,cov_num)
if pnu_num > chk_num:
    del pneumonia_cases[:pnu_num-chk_num]
if nor_num > chk_num:
    del healthy_cases[:nor_num-chk_num]
if cov_num > chk_num:
    del covid_cases[:cov_num-chk_num]

num = len(healthy_cases)
train_num = math.floor(num/100*70)
test_num = math.floor(num/100*15)
val_num = math.floor(num/100*15)
r = num-(train_num+test_num+val_num)
train_num+=r

train_healthy_cases = healthy_cases[:train_num]
train_pneumonia_cases = pneumonia_cases[:train_num]
train_covid_cases = covid_cases[:train_num]
del healthy_cases[:train_num]
del pneumonia_cases[:train_num]
del covid_cases[:train_num]

test_healthy_cases = healthy_cases[:test_num]
test_pneumonia_cases = pneumonia_cases[:test_num]
test_covid_cases = covid_cases[:test_num]
del healthy_cases[:test_num]
del pneumonia_cases[:test_num]
del covid_cases[:test_num]

val_healthy_cases = healthy_cases[:val_num]
val_pneumonia_cases = pneumonia_cases[:val_num]
val_covid_cases = covid_cases[:val_num]

train_list = []
test_list = []
val_list = []

j=0 
for i in train_healthy_cases:
    # experiment.log_image(i, name='training_healthy_'+str(j), overwrite=False)
    train_list.append([i, 0])
    j+=1

k=0
for i in train_pneumonia_cases:
    # experiment.log_image(i, name='training_pneumonia_'+str(j), overwrite=False)
    train_list.append([i, 1])
    k+=1
    if k == j:
        break
k=0
for i in train_covid_cases:
    # experiment.log_image(i, name='training_pneumonia_'+str(j), overwrite=False)
    train_list.append([i, 2])
    k+=1
    if k == j:
        break


j=0
for i in test_healthy_cases:
    # experiment.log_image(i, name='testing_healthy_'+str(j), overwrite=False)
    test_list.append([i, 0])
    j+=1

k=0
for i in test_pneumonia_cases:
    # experiment.log_image(i, name='testing_pneumonia_'+str(j), overwrite=False)
    test_list.append([i, 1])
    k+=1
    if k == j:
        break
k=0
for i in test_covid_cases:
    # experiment.log_image(i, name='testing_pneumonia_'+str(j), overwrite=False)
    test_list.append([i, 2])
    k+=1
    if k == j:
        break

j=0
for i in val_healthy_cases:
    # experiment.log_image(i, name='validation_healthy_'+str(j), overwrite=False)
    val_list.append([i, 0])
    j+=1

k=0
for i in val_pneumonia_cases:
    # experiment.log_image(i, name='validation_pneumonia_'+str(j), overwrite=False)
    val_list.append([i, 1])
    k+=1
    if k == j:
        break
k=0
for i in val_covid_cases:
    # experiment.log_image(i, name='validation_pneumonia_'+str(j), overwrite=False)
    val_list.append([i, 2])
    k+=1
    if k == j:
        break

## shuffle data
rnd.shuffle(train_list)
rnd.shuffle(test_list)
rnd.shuffle(val_list)

#create dataframes
train_df = pd.DataFrame(train_list, columns=['image','label'])
test_df = pd.DataFrame(test_list, columns=['image','label'])
val_df = pd.DataFrame(val_list, columns=['image','label'])

print("\n\nDataframes:")
print("___________________________________________________________________________________________")
print(train_df)
print("___________________________________________________________________________________________")
print("___________________________________________________________________________________________")
print(test_df)
print("___________________________________________________________________________________________")
print("___________________________________________________________________________________________")
print(val_df)
print("___________________________________________________________________________________________")

visualise = 'n'
if visualise == 'y':
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
        plt.title('pneumonia')
    
    for i,img_path in enumerate(train_df[train_df['label'] == 0][0:4]['image']):
        plt.subplot(2,4,4+i+1)
        plt.axis('off')
        img = plt.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title('healthy')

    plt.show()

print('Data visualisation complete')
# cont = input('Enter to continue')

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
y_train = to_categorical(y_train,3)
y_test = to_categorical(y_test,3)
y_val = to_categorical(y_val,3)

class ConfusionMatrixCallback(Callback):
    def __init__(self, experiment, inputs, targets):
        self.experiment = experiment
        self.inputs = inputs
        self.targets = targets

    def on_epoch_end(self, epoch, logs={}):
        predicted = self.model.predict(self.inputs)
        self.experiment.log_confusion_matrix(
            self.targets,
            predicted,
            title="Confusion Matrix, Epoch #%d" % (epoch + 1),
            file_name="confusion-matrix-%03d.json" % (epoch + 1),
        )

# @tf.function
# def custom_sigmoid(x):
#     print(x)
#     print("---------")
#     x = (1/float(1+np.exp(-x)))-param_vals["model"]["threshold"]
#     print(x)
#     return x
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
model.add(Dense(3, activation="softmax"))

if param_vals['optimiser']['name'] == "Adam":
    params = param_vals['optimiser']
    optimizer = Adam(lr=params['lr'], beta_1=params['beta_1'], beta_2=params['beta_2'], epsilon=params['epsilon'], decay=params['decay'])
params = param_vals['compile']
model.compile(loss=params['loss'], optimizer=optimizer, metrics=params['metrics'], loss_weights=params['loss_weights'], weighted_metrics=params['weighted_metrics'], run_eagerly=params['run_eagerly'])

callback = [ConfusionMatrixCallback(experiment, X_test, y_test),EarlyStopping(monitor='loss', patience=6)]
history = model.fit(datagen.flow(X_train,y_train, batch_size=4), validation_data=(X_test, y_test), epochs = param_vals["model"]["epoch"], verbose = 1, callbacks=[callback[0]], class_weight=[{0:6.0, 1:0.5, 2:0.5}])

################################################################################################################################
##Evaluation

plt.figure(figsize = (20,5))

plt.subplot(1,2,1)
sb.lineplot(x = history.epoch, y = history.history['loss'], color='red', label = 'Loss')
sb.lineplot(x = history.epoch, y = history.history['val_loss'], color='orange', label='Validation Loss')
plt.title('Loss on training vs testing')
plt.legend(loc = 'best')
plt.xlabel("Epoch #")
plt.ylabel("Loss")

plt.subplot(1,2,2)
sb.lineplot(x = history.epoch, y = history.history['accuracy'], color = 'blue', label = 'Accuracy')
sb.lineplot(x = history.epoch, y = history.history['val_accuracy'], color = 'green', label = 'Validation Accuracy')
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.title('Accuracy on training vs testing')
plt.legend(loc = 'best')
plt.show()

for i in history.epoch:
    experiment.log_metric("accuracy", history.history['accuracy'][i], epoch=i+1)
    experiment.log_metric("loss", history.history['loss'][i],epoch=i+1)

################################################################################################################################
## Confusion matrix

y_test_hat = model.predict(X_test, batch_size=4)
y_test_hat = np.argmax(y_test_hat, axis=1)
y_test = np.argmax(y_test, axis=1)
experiment.log_confusion_matrix(y_test,y_test_hat)
conf_m = confusion_matrix(y_test, y_test_hat)
print(conf_m)
clas_r = classification_report(y_test, y_test_hat)

plt.figure(figsize = (5,3))
sb.set(font_scale = 1.2)
ax = sb.heatmap(conf_m, annot=True, xticklabels = ['N','P'], yticklabels = ['N', 'P'], cbar=False, cmap='Blues', linewidths=1, linecolor='black', fmt='.0f')
plt.yticks(rotation=0)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
ax.xaxis.set_ticks_position('top')
plt.title('Confusion matrix - (N - healthy, P - pneumonia):')
plt.show()

################################################################################################################################
## Validation & Report

print('Classification report on test data')
print(clas_r)

y_val_hat = model.predict(X_val, batch_size=4)
y_val_hat = np.argmax(y_val_hat, axis=1)
y_val = np.argmax(y_val, axis=1)

## Summary
print(model.summary())

if param_vals["model"]["save"]==True:
    print("saving?")
    # model.save("model")
    # tf.saved_model.save(model,"saved_model")
    save_model(model,"save_model")

if __name__ != '__main__':
    ret["Data"]={}
    for book in history.history.keys():
        ret["Data"][book]=[float(history.history[book][len(history.epoch)-1])]
    for shelf in param_vals:
        ret[shelf]={}
        print("Storing: ",shelf)
        for book,content in param_vals[shelf].items():
            ret[shelf][book]=[content]
    conf_m = conf_m.tolist()
    ret["Data"]["conf_matrix"]=[conf_m]

    # ret.append(["loss"+str(param_vals["iteration"]["#"]),[history.history['loss'][len(history.epoch)-1]]])
    # ret.append(["lr",[param_vals['optimiser']['lr']],False])

    comm.send(ret, dest=0, tag=comm.Get_rank())
else:
    print("----------------------")
    print("Storing: History")
    for book in history.history.keys():
        json_comm.store_data(book,[float(history.history[book][len(history.epoch)-1])],"Data")
    for shelf in param_vals:
        print("Storing: ",shelf)
        for book,content in param_vals[shelf].items():
            json_comm.store_data(str(book),[content],shelf,False)
    print(conf_m[0])
    conf_m = conf_m.tolist()
    print(conf_m)
    json_comm.store_data("conf_matrix",[conf_m],"Data")
    # json_comm.verify_data_length()

