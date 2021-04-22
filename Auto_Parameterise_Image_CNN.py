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
import visualkeras
from collections import defaultdict
from PIL import ImageFont

font = ImageFont.truetype("arial.ttf", 32)
color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'orange'
color_map[Dropout]['fill'] = 'green'
color_map[MaxPooling2D]['fill'] = 'red'
color_map[Dense]['fill'] = 'teal'
color_map[Flatten]['fill'] = 'gray'

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

val_negative_cases = []
val_positive_cases = []

## Using glob acts as a file type filter to ensure type syncronosity
positive_cases = glob.glob(path + param_vals["model"]["data"][1] + '/*.jpeg')
negative_cases = glob.glob(path + param_vals["model"]["data"][0] + '/*.jpeg')
rnd.shuffle(positive_cases)
rnd.shuffle(negative_cases)

pos_num = len(positive_cases)
neg_num = len(negative_cases)
print(pos_num)
print(neg_num)
chk_num=pos_num-neg_num
print(chk_num)
if chk_num > 0:
    del positive_cases[:abs(chk_num)]
elif chk_num < 0:
    del negative_cases[:abs(chk_num)]
num = len(positive_cases)

train_num = math.floor(num/100*70)
test_num = math.floor(num/100*15)
val_num = math.floor(num/100*15)
r = num-(train_num+test_num+val_num)
train_num+=r

train_negative_cases = negative_cases[:train_num]
train_positive_cases = positive_cases[:train_num]
del negative_cases[:train_num]
del positive_cases[:train_num]
# print("----------------------------------------------")
# print(len(train_negative_cases)," elements:\n",train_negative_cases)
# print("----------------------------------------------")
# print(len(train_positive_cases)," elements:\n",train_positive_cases)
test_negative_cases = negative_cases[:test_num]
test_positive_cases = positive_cases[:test_num]
del negative_cases[:test_num]
del positive_cases[:test_num]
# print("----------------------------------------------")
# print(len(test_negative_cases)," elements:\n",test_negative_cases)
# print("----------------------------------------------")
# print(len(test_positive_cases)," elements:\n",test_positive_cases)
val_negative_cases = negative_cases[:val_num]
val_positive_cases = positive_cases[:val_num]
# print("----------------------------------------------")
# print(len(val_negative_cases)," elements:\n",val_negative_cases)
# print("----------------------------------------------")
# print(len(val_positive_cases)," elements:\n",val_positive_cases)
train_list = []
test_list = []
val_list = []

j=0 
for i in train_negative_cases:
    # experiment.log_image(i, name='training_negative_'+str(j), overwrite=False)
    train_list.append([i, 0])
    j+=1

k=0
for i in train_positive_cases:
    # experiment.log_image(i, name='training_positive_'+str(j), overwrite=False)
    train_list.append([i, 1])
    k+=1
    if k == j:
        break

j=0
for i in test_negative_cases:
    # experiment.log_image(i, name='testing_negative_'+str(j), overwrite=False)
    test_list.append([i, 0])
    j+=1

k=0
for i in test_positive_cases:
    # experiment.log_image(i, name='testing_positive_'+str(j), overwrite=False)
    test_list.append([i, 1])
    k+=1
    if k == j:
        break

j=0
for i in val_negative_cases:
    # experiment.log_image(i, name='validation_negative_'+str(j), overwrite=False)
    val_list.append([i, 0])
    j+=1

k=0
for i in val_positive_cases:
    # experiment.log_image(i, name='validation_positive_'+str(j), overwrite=False)
    val_list.append([i, 1])
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
    
    # plt.show()

#sample images
    plt.figure(figsize=(20,8))
    for i,img_path in enumerate(train_df[train_df['label'] == 1][0:4]['image']):
        plt.subplot(2,4,i+1)
        plt.axis('off')
        img = plt.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title('Positive')
    
    for i,img_path in enumerate(train_df[train_df['label'] == 0][0:4]['image']):
        plt.subplot(2,4,4+i+1)
        plt.axis('off')
        img = plt.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title('Negative')

    # plt.show()

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
    zoom_range = [0.9,1.2],
    width_shift_range = 0.1,
    horizontal_flip = False,
    vertical_flip = False)

datagen.fit(X_train)

# Convert binary to categorical (process time improvement)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

class ConfusionMatrixCallback(Callback):
    def __init__(self, experiment, inputs, targets):
        self.experiment = experiment
        self.inputs = inputs
        self.targets = targets

    def on_epoch_end(self, epoch, logs={}):
        predicted = self.model.predict(self.inputs)

        # y_test_hat = np.argmax(predicted, axis=1)
        # y_test = np.argmax(self.targets, axis=1)
        # conf_m = confusion_matrix(y_test, y_test_hat)
        # conf_m = conf_m.tolist()
        # json_comm.pack_matrix(conf_m)

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

model.add(Conv2D(filters=8, kernel_size=(7,7), padding='same', activation='relu', name="activation_1", input_shape=(196, 196, 1)))
model.add(Conv2D(filters=8, kernel_size=(7,7), padding='same', activation='relu', name="activation_2"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(visualkeras.SpacingDummyLayer(spacing=100))
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu', name="activation_3"))
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu', name="activation_4"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(visualkeras.SpacingDummyLayer(spacing=100))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', name="activation_5"))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', name="activation_6"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(visualkeras.SpacingDummyLayer(spacing=100))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', name="activation_7"))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', name="activation_8"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(visualkeras.SpacingDummyLayer(spacing=100))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name="activation_9"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name="activation_10"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(visualkeras.SpacingDummyLayer(spacing=100))
model.add(Flatten())
model.add(visualkeras.SpacingDummyLayer(spacing=100))
model.add(Dense(128, activation='relu', name="activation_11"))
model.add(Dropout(0.4))
model.add(Dense(2, activation="sigmoid", name="activation_12"))

if param_vals['optimiser']['name'] == "Adam":
    params = param_vals['optimiser']
    optimizer = Adam(lr=params['lr'], beta_1=params['beta_1'], beta_2=params['beta_2'], epsilon=params['epsilon'], decay=params['decay'])
params = param_vals['compile']
model.compile(loss=params['loss'], optimizer=optimizer, metrics=params['metrics'], loss_weights=params['loss_weights'], weighted_metrics=params['weighted_metrics'], run_eagerly=params['run_eagerly'])

callback = [ConfusionMatrixCallback(experiment, X_test, y_test),
EarlyStopping(monitor=str(param_vals["callback"]["EarlyStopping"]["monitor"]), patience=param_vals["callback"]["EarlyStopping"]["patience"]),
tf_cb.GradCAMCallback(
        validation_data=(X_test[0:4],y_test[0:4]),
        layer_name="activation_2",
        class_index=0,
        output_dir="./summaries_0",
    ),
tf_cb.GradCAMCallback(
        validation_data=(X_test[0:4],y_test[0:4]),
        layer_name="activation_2",
        class_index=1,
        output_dir="./summaries_1",
    ),
tf_cb.GradCAMCallback(
        validation_data=(X_test[0:4],y_test[0:4]),
        layer_name="activation_4",
        class_index=0,
        output_dir="./summaries_0",
    ),
tf_cb.GradCAMCallback(
        validation_data=(X_test[0:4],y_test[0:4]),
        layer_name="activation_4",
        class_index=1,
        output_dir="./summaries_1",
    ),
tf_cb.GradCAMCallback(
        validation_data=(X_test[0:4],y_test[0:4]),
        layer_name="activation_6",
        class_index=0,
        output_dir="./summaries_0",
    ),
tf_cb.GradCAMCallback(
        validation_data=(X_test[0:4],y_test[0:4]),
        layer_name="activation_6",
        class_index=1,
        output_dir="./summaries_1",
    ),
tf_cb.GradCAMCallback(
        validation_data=(X_test[0:4],y_test[0:4]),
        layer_name="activation_8",
        class_index=0,
        output_dir="./summaries_0",
    ),
tf_cb.GradCAMCallback(
        validation_data=(X_test[0:4],y_test[0:4]),
        layer_name="activation_8",
        class_index=1,
        output_dir="./summaries_1",
    ),
tf_cb.GradCAMCallback(
        validation_data=(X_test[0:4],y_test[0:4]),
        layer_name="activation_10",
        class_index=0,
        output_dir="./summaries_0",
    ),
tf_cb.GradCAMCallback(
        validation_data=(X_test[0:4],y_test[0:4]),
        layer_name="activation_10",
        class_index=1,
        output_dir="./summaries_1",
    )
]

history = model.fit(datagen.flow(X_train,y_train, batch_size=4), validation_data=(X_test, y_test), epochs = param_vals["model"]["epoch"], verbose = 1, callbacks=callback, class_weight=[{0:param_vals["model"]["class_weights"][0], 1:param_vals["model"]["class_weights"][1]}])
# history = model.fit(datagen.flow(X_train,y_train, batch_size=4), validation_data=(X_test, y_test), epochs = param_vals["model"]["epoch"], verbose = 1, callbacks=[callback[0]], class_weight=[{0:6.0, 1:0.5}])

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
# plt.show()

for i in history.epoch:
    experiment.log_metric("accuracy", history.history['accuracy'][i], epoch=i+1)
    experiment.log_metric("loss", history.history['loss'][i],epoch=i+1)

################################################################################################################################
## Confusion matrix
visualkeras.layered_view(model,color_map=color_map,spacing=0,legend=True, font=font, to_file='two_model_layered_output.png').show() # write and show
visualkeras.graph_view(model, to_file='two_model_graph_output.png').show() # write and show
y_test_hat = model.predict(X_test, batch_size=4)
y_test_hat = np.argmax(y_test_hat, axis=1)
y_test = np.argmax(y_test, axis=1)
experiment.log_confusion_matrix(y_test,y_test_hat)
conf_m = confusion_matrix(y_test, y_test_hat)
print(conf_m)
clas_r = classification_report(y_test, y_test_hat)

conf_m = confusion_matrix(y_test, y_test_hat)
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
    # json_comm.pack_un()

