#Support vector machine for fall detection

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('voice.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,20].values

#labelencode the data
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

#Transforming the dataset into training and testing dataset
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Fitting SVM to the training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',C = 8,gamma = 0.1,random_state = 0)
classifier.fit(X_train,y_train)

#Fitting the test set results
y_pred = classifier.predict(X_test)

#Making the confusing matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


#Making the model more accurate by importing k cross validation and grid search
#Applying k fold cross validation

from sklearn.model_selection import cross_val_score 
accuracies = cross_val_score(estimator = classifier, X= X_train,y = y_train, cv = 10,n_jobs = -1)
mean = accuracies.mean()
standard_deviation=accuracies.std() #Average of the differences between the different accurate we get from different dataset in the model is 0.01 meaning it is not very high variance

#Gridsearch (Improving the model performance by finding the hyperparameters)
#Applying gridsearch to find the best models and the best parameters

from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1,10,100,1000], 'kernel':['linear']}, #C is for the regularization
              {'C': [1,2,3,4,5,6,7,8,9,10,100,1000], 'kernel': ['rbf'],'gamma':[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.5,0.1,0.001,0.001]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_