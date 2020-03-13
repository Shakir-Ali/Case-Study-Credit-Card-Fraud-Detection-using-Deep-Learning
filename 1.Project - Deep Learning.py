import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
data = pd.read_csv('creditcard.csv')

## Data Preprocessing

from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Time', 'Amount'], axis = 1)

x = data.drop(['Class'], axis = 1)
y = data['Class']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

## Deep Neural Networks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential([
    Dense(units = 16, input_dim = 29, activation = 'relu'),
    Dense(units = 24, activation = 'relu'),
    Dropout(0.5),
    Dense(units = 20, activation = 'relu'),
    Dense(units = 24, activation = 'relu'),
    Dense(1, activation = 'sigmoid')])
#print(model.summary())

## Training
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 15, epochs = 5)

score = model.evaluate(x_test, y_test)
print(score)

## Confusion Matrix
y_pred = model.predict(x_test)
y_test = pd.DataFrame(y_test)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred.round())
print(cm)
sns.heatmap(cm, annot = True)
#plt.show()
