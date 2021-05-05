from comet_ml import Experiment, ConfusionMatrix
experiment = Experiment(
    api_key="",
    project_name="",
    workspace="",
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
import tf_explain.callbacks as tf_cb
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

chk_num=min(pnu_num,nor_num,cov_num)
if pnu_num > chk_num:
    del pneumonia_cases[:pnu_num-abs(chk_num)]
if nor_num > chk_num:
    del healthy_cases[:nor_num-abs(chk_num)]
if cov_num > chk_num:
    del covid_cases[:cov_num-abs(chk_num)]

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

# Compose data lists ensuring balanced numbers of data.
train_list = []
test_list = []
val_list = []


for i in train_healthy_cases:
    train_list.append([i, 0])
for i in train_pneumonia_cases:
    train_list.append([i, 1])
for i in train_covid_cases:
    train_list.append([i, 2])
for i in test_healthy_cases:
    test_list.append([i, 0])
for i in test_pneumonia_cases:
    test_list.append([i, 1])
for i in test_covid_cases:
    test_list.append([i, 2])
for i in val_healthy_cases:
    val_list.append([i, 0])
for i in val_pneumonia_cases:
    val_list.append([i, 1])
for i in val_covid_cases:
    val_list.append([i, 2])

# Shuffle data
rnd.shuffle(train_list)
rnd.shuffle(test_list)
rnd.shuffle(val_list)

# Create dataframes
train_df = pd.DataFrame(train_list, columns=['image','label'])
test_df = pd.DataFrame(test_list, columns=['image','label'])
val_df = pd.DataFrame(val_list, columns=['image','label'])

# switch variable if using task-farm-cnn.py
# visualise = input("Visualise data? y/n").lower()
visualise = 'n'
if visualise == 'y':

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

    # Display data distribution
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

    # Display sample images
    plt.figure(figsize=(20,8))
    for i,img_path in enumerate(train_df[train_df['label'] == 2][0:4]['image']):
        plt.subplot(3,4,i+1)
        plt.axis('off')
        img = plt.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title('Covid-19')

    for i,img_path in enumerate(train_df[train_df['label'] == 1][0:4]['image']):
        plt.subplot(3,4,4+i+1)
        plt.axis('off')
        img = plt.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title('Pneumonia')
    
    for i,img_path in enumerate(train_df[train_df['label'] == 0][0:4]['image']):
        plt.subplot(3,4,8+i+1)
        plt.axis('off')
        img = plt.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title('Healthy')

    plt.show()

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

# Data generation
datagen = ImageDataGenerator(
    rotation_range = 10,
    zoom_range = [0.9,1.2],
    width_shift_range = 0.1,
    horizontal_flip = False,
    vertical_flip = False)

datagen.fit(X_train)

# Convert binary to categorical (process time improvement)
y_train = to_categorical(y_train,3)
y_test = to_categorical(y_test,3)
y_val = to_categorical(y_val,3)

#############################################################################################################################
## Model Design

# Build Model
model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(7,7), padding='same', activation='relu', input_shape=(196, 196, 1)))
model.add(Conv2D(filters=8, kernel_size=(7,7), padding='same', activation='relu', name="activation_2"))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu', name="activation_4"))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', name="activation_6"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', name="activation_8"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name="activation_10"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(3, activation="softmax"))

if param_vals['optimiser']['name'] == "Adam":
    params = param_vals['optimiser']
    optimizer = Adam(lr=params['lr'], beta_1=params['beta_1'], beta_2=params['beta_2'], epsilon=params['epsilon'], decay=params['decay'])
params = param_vals['compile']

model.compile(loss=params['loss'], optimizer=optimizer, metrics=params['metrics'], loss_weights=params['loss_weights'], weighted_metrics=params['weighted_metrics'], run_eagerly=params['run_eagerly'])

# Custom callback function for passing multiple confusion matrices to Comet-ML
class ConfusionMatrixCallback(Callback):
    def __init__(self, experiment, inputs, targets):
        self.experiment = experiment
        self.inputs = inputs
        self.targets = targets

    def on_epoch_end(self, epoch, logs={}):
        predicted = self.model.predict(self.inputs)

        ## Store current epoch confusion matrix
        if param_vals["callback"]["Pack_Matrix"] == True:
            y_test_hat = np.argmax(predicted, axis=1)
            y_test = np.argmax(self.targets, axis=1)
            conf_m = confusion_matrix(y_test, y_test_hat)
            conf_m = conf_m.tolist()
            json_comm.pack_matrix(conf_m)

        self.experiment.log_confusion_matrix(
            self.targets,
            predicted,
            title="Confusion Matrix, Epoch #%d" % (epoch + 1),
            file_name="confusion-matrix-%03d.json" % (epoch + 1),
        )

# Define callback functions for use during training.
callback = [ConfusionMatrixCallback(experiment, X_test, y_test),
            EarlyStopping(monitor=str(param_vals["callback"]["EarlyStopping"]["monitor"]), patience=param_vals["callback"]["EarlyStopping"]["patience"]),
            tf_cb.GradCAMCallback(
                    validation_data=(X_test[0:4],y_test[0:4]),
                    layer_name="activation_2",
                    class_index=0,
                    output_dir="./3_class_summaries_0",
                ),
            tf_cb.GradCAMCallback(
                    validation_data=(X_test[0:4],y_test[0:4]),
                    layer_name="activation_2",
                    class_index=1,
                    output_dir="./3_class_summaries_1",
                ),
            tf_cb.GradCAMCallback(
                    validation_data=(X_test[0:4],y_test[0:4]),
                    layer_name="activation_2",
                    class_index=2,
                    output_dir="./3_class_summaries_2",
                ),
            tf_cb.GradCAMCallback(
                    validation_data=(X_test[0:4],y_test[0:4]),
                    layer_name="activation_4",
                    class_index=0,
                    output_dir="./3_class_summaries_0",
                ),
            tf_cb.GradCAMCallback(
                    validation_data=(X_test[0:4],y_test[0:4]),
                    layer_name="activation_4",
                    class_index=1,
                    output_dir="./3_class_summaries_1",
                ),
            tf_cb.GradCAMCallback(
                    validation_data=(X_test[0:4],y_test[0:4]),
                    layer_name="activation_4",
                    class_index=2,
                    output_dir="./3_class_summaries_2",
                ),
            tf_cb.GradCAMCallback(
                    validation_data=(X_test[0:4],y_test[0:4]),
                    layer_name="activation_6",
                    class_index=0,
                    output_dir="./3_class_summaries_0",
                ),
            tf_cb.GradCAMCallback(
                    validation_data=(X_test[0:4],y_test[0:4]),
                    layer_name="activation_6",
                    class_index=1,
                    output_dir="./3_class_summaries_1",
                ),
            tf_cb.GradCAMCallback(
                    validation_data=(X_test[0:4],y_test[0:4]),
                    layer_name="activation_6",
                    class_index=2,
                    output_dir="./3_class_summaries_2",
                ),
            tf_cb.GradCAMCallback(
                    validation_data=(X_test[0:4],y_test[0:4]),
                    layer_name="activation_8",
                    class_index=0,
                    output_dir="./3_class_summaries_0",
                ),
            tf_cb.GradCAMCallback(
                    validation_data=(X_test[0:4],y_test[0:4]),
                    layer_name="activation_8",
                    class_index=1,
                    output_dir="./3_class_summaries_1",
                ),
            tf_cb.GradCAMCallback(
                    validation_data=(X_test[0:4],y_test[0:4]),
                    layer_name="activation_8",
                    class_index=2,
                    output_dir="./3_class_summaries_2",
                ),
            tf_cb.GradCAMCallback(
                    validation_data=(X_test[0:4],y_test[0:4]),
                    layer_name="activation_10",
                    class_index=0,
                    output_dir="./3_class_summaries_0",
                ),
            tf_cb.GradCAMCallback(
                    validation_data=(X_test[0:4],y_test[0:4]),
                    layer_name="activation_10",
                    class_index=1,
                    output_dir="./3_class_summaries_1",
                ),
            tf_cb.GradCAMCallback(
                    validation_data=(X_test[0:4],y_test[0:4]),
                    layer_name="activation_10",
                    class_index=2,
                    output_dir="./3_class_summaries_2",
                )
            ]

# Train model
history = model.fit(datagen.flow(X_train,y_train, batch_size=4), validation_data=(X_test, y_test), epochs = param_vals["model"]["epoch"], verbose = 1, callbacks=callback, class_weight=[{0:param_vals["model"]["three_class_weights"][0], 1:param_vals["model"]["three_class_weights"][1], 2:param_vals["model"]["three_class_weights"][2]}])

################################################################################################################################
## Evaluation

# Plot Epoch vs Accuracy/Loss
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

# Comet-ML log data
for i in history.epoch:
    experiment.log_metric("accuracy", history.history['accuracy'][i], epoch=i+1)
    experiment.log_metric("loss", history.history['loss'][i],epoch=i+1)


## Confusion matrix
y_test_hat = model.predict(X_test, batch_size=4)
y_test_hat = np.argmax(y_test_hat, axis=1)
y_test = np.argmax(y_test, axis=1)
conf_m = confusion_matrix(y_test, y_test_hat)

# Comet-ML log data
experiment.log_confusion_matrix(y_test,y_test_hat)

# Plot matrix
plt.figure(figsize = (5,3))
sb.set(font_scale = 1.2)
ax = sb.heatmap(conf_m, annot=True, xticklabels = ['N','P'], yticklabels = ['N', 'P'], cbar=False, cmap='Blues', linewidths=1, linecolor='black', fmt='.0f')
plt.yticks(rotation=0)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
ax.xaxis.set_ticks_position('top')
plt.title('Confusion matrix - (N - Negative, P - Positive):')

plt.show()

################################################################################################################################
## Validation & Report

clas_r = classification_report(y_test, y_test_hat)
print('Classification report')
print(clas_r)

## Summary
print(model.summary())

################################################################################################################################
## Save Model/Data

# Save model to directory
if param_vals["model"]["save"]==True:
    save_model(model,"save_model_3_class")

if __name__ != '__main__':
    # Slave: Organise data, send to master thread.
    ret["Data"]={}
    for book in history.history.keys():
        ret["Data"][book]=[float(history.history[book][len(history.epoch)-1])]
    for shelf in param_vals:
        ret[shelf]={}
        for book,content in param_vals[shelf].items():
            ret[shelf][book]=[content]
    conf_m = conf_m.tolist()
    ret["Data"]["conf_matrix"]=[conf_m]
    comm.send(ret, dest=0, tag=comm.Get_rank())

else:
    # Pass data to json_comm for saving to output json.
    for book in history.history.keys():
        json_comm.store_data(book,[float(history.history[book][len(history.epoch)-1])],"Data")
    for shelf in param_vals:
        for book,content in param_vals[shelf].items():
            json_comm.store_data(str(book),[content],shelf,False)
    conf_m = conf_m.tolist()
    json_comm.store_data("conf_matrix",[conf_m],"Data")