# import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
import numpy as np
import random as rnd
import cv2
import glob
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical

loaded_model=load_model("save_model", compile = True)
positive_cases = glob.glob('./active_data/CV/*.jpeg')
negative_cases = glob.glob('./active_data/N/*.jpeg')
rnd.shuffle(positive_cases)
rnd.shuffle(negative_cases)

test_list = []
j=0 
for i in negative_cases:
    # experiment.log_image(i, name='training_negative_'+str(j), overwrite=False)
    test_list.append([i, 0])
    j+=1

k=0
for i in positive_cases:
    # experiment.log_image(i, name='training_positive_'+str(j), overwrite=False)
    test_list.append([i, 1])
    k+=1
    if k == j:
        break
print(len(negative_cases))
print(len(positive_cases))
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

test_df = pd.DataFrame(test_list, columns=['image','label'])
print(test_df)
X_test, y_test = compose_dataset(test_df)
y_test = to_categorical(y_test)

y_test_hat = loaded_model.predict(X_test, batch_size=4)
y_test_hat = np.argmax(y_test_hat, axis=1)
y_test = np.argmax(y_test, axis=1)

conf_m = confusion_matrix(y_test, y_test_hat)
clas_r = classification_report(y_test, y_test_hat)

plt.figure(figsize = (5,3))
sb.set(font_scale = 1.2)
ax = sb.heatmap(conf_m, annot=True, xticklabels = ['Negative','Positive'], yticklabels = ['Negative', 'Positive'], cbar=False, cmap='Blues', linewidths=1, linecolor='black', fmt='.0f')
plt.yticks(rotation=0)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
ax.xaxis.set_ticks_position('top')
plt.title('Confusion matrix - test data\n(N - Negative, P - Positive)')

plt.show()

