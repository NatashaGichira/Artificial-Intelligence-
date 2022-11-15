"""###### CLASSIFICATION ALGORITHMS ######"""

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# Regression
from sklearn.linear_model import LinearRegression

# Load the data from the Iris.csv file using pandas (relative path as csv should be in same folder as this code)
dataset = pd.read_csv('Iris.csv')

# Specify the column names in the dataset that are features
feature_column_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Values in each feature column is stored in variable 'X'
X = dataset[feature_column_names].values

# Values in 'Species' column is stored in variable 'y'. THESE ARE OUR LABELS
y = dataset['Species'].values

# Encode the different labels in the species column eg: 0,1,2...
y = LabelEncoder().fit_transform(y)

# Split the dataset into 60% for training and 40% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

# Fit the training features 'X_train' to the training labels 'y_train' using the KNN Classifier model
knn = KNeighborsClassifier().fit(X_train, y_train)

# After fitting using the training data, make predictions on the correct label based on test features 'X_test'
knn_prediction = knn.predict(X_test)

# Fit the training features 'X_train' to the training labels 'y_train' using the Decision Tree Classifier model
dt = DecisionTreeClassifier().fit(X_train, y_train)

# After fitting using the training data, make predictions on the correct label based on test features 'X_test'
dt_prediction = dt.predict(X_test)

# Calculate Accuracy of KNN Model
knn_accuracy = accuracy_score(y_test, knn_prediction)*100
print('Accuracy of KNN model: ' + str(knn_accuracy) + ' %')

# Calculate Accuracy of Decision Tree Model
dt_accuracy = accuracy_score(y_test, dt_prediction)*100
print('Accuracy of Decision Tree model: ' + str(dt_accuracy) + ' %')

"""###### LINEAR REGRESSION MODEL ######"""

# Set 'SepalLengthCm' as X and 'PetalLengthCm' as Y for our Linear Regression Model and plot their relationship
lr_X = dataset['PetalLengthCm']
lr_Y = dataset['SepalLengthCm']

plt.figure(figsize=(16,6))
plt.scatter(lr_X, lr_Y)
plt.xlabel('PetalLength(cm)')
plt.ylabel('SepalLength(cm)')
plt.title('Petal/Sepal Length Relationship')
plt.show()

# Reshape the X into a 2D array that can be accepted by the LinearRegression function
lr_X = np.array(lr_X).reshape(-1,1)

# Split the dataset into 60% for training and 40% for testing
lr_X_train, lr_X_test, lr_Y_train, lr_Y_test = train_test_split(lr_X, lr_Y, test_size = 0.40)

# Train the LR model on the training data
lr = LinearRegression().fit(lr_X_train, lr_Y_train)

# Plotting the regression line
line = lr.coef_ * lr_X + lr.intercept_

plt.figure(figsize=(16,6))
plt.plot(lr_X, line, color='red')
plt.scatter(lr_X, lr_Y, color='blue')
plt.xlabel('PetalLength(cm)')
plt.ylabel('SepalLength(cm)')
plt.title('Regression Line')
plt.show()

# Calculate Accuracy of Model
lr_predictions = lr.predict(lr_X_test)

print('Accuracy: ' + str(lr.score(lr_X_test, lr_Y_test) * 100) + '%')
