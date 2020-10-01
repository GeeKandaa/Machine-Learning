## Introductory machine learning program in python

# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# Following the tutorial at the above address
##

## Libraries

# Panda
from pandas import read_csv
from pandas.plotting import scatter_matrix
# MatPlotLib
from matplotlib import pyplot
# SciKit-Learn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset = read_csv(url,names=names)
# check dataset
print('\n\nDataset preview:\n',dataset.head(5),'\n...\n')
print('Dataset shape: ',dataset.shape)
print('\n\nDataset summary:\n',dataset.describe())
print('\n\nDataset class sizes:\n',dataset.groupby('class').size())

# dataset input attribute visualisation
print('\n\nInput data visualisation:\n')
print('\nBox and Whisker:\n[graphic]\n')
dataset.plot(kind='box', subplots=True, layout=(1,4), sharex=False, sharey=True)
pyplot.show()
print('\nHistogram:\n[graphic]\n')
dataset.hist()
pyplot.show()
print('\nScatter Plot:\n[graphic]\n')
scatter_matrix(dataset)
pyplot.show()

# Split dataset into training and validation partitions
array = dataset.values
x = array[:,0:4]
y = array[:,4]
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=1)

# debug print values
#print('x train:',x_train)
#print('y train:',y_train)
#print('x validation',x_validation)
#print('y validation',y_validation)

## Spot check algorithms
# Prepare list of models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluation of models
results=[]
names=[]
print('\nEvaluation of Models:')
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Algrotihm comparison
print('\nAlgorithm Comparison:')
print('\nBox and Whisker:\n[graphic]\n')
pyplot.boxplot(results, labels=names)
pyplot.show()


## Make predictions on validation dataset
# Algorithm comparison suggests SVC as most accurate model
print('\nValidation using SVC model')
model = SVC(gamma='auto')
model.fit(x_train,y_train)
predictions = model.predict(x_validation)
print('\nAccuracy score:\n', accuracy_score(y_validation, predictions))
print('\nConfusion matrix:\n', confusion_matrix(y_validation, predictions))
print('\nClassification report:\n', classification_report(y_validation, predictions))
