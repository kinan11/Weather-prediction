import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def build_model():
    model = Sequential()
    model.add(Dense(150, input_dim=32, activation='sigmoid'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    return model


build_model().summary()

data = pd.read_csv('Dataset.csv')
data = data.drop('Dzień', axis=1)
data = data.drop('Miesiąc', axis=1)
data = data.drop('Rok', axis=1)

# Pozyskanie danych
x_tmp = data.iloc[:, 1:].values
y = data.iloc[4:, 3].values

x = []
for i in range(len(x_tmp) - 4):
    x.append([])
    x[i].extend(x_tmp[i])
    x[i].extend(x_tmp[i + 1])
    x[i].extend(x_tmp[i + 2])
    x[i].extend(x_tmp[i + 3])

print(len(x[0]))

# Podział na dane uczące i trenujące
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Skalowanie danych
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

model = build_model()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

ia = (1 - (np.sum((y_test - y_pred) ** 2)) / (
    np.sum((np.abs(y_pred - np.mean(y_test)) + np.abs(y_test - np.mean(y_test))) ** 2)))
print(str(y_pred[0]) + " " + str(y_test[0]))

# print(len(build_model().get_weights()[1][0]))

print('Neural Network: ')
print('Mean squared error: ', mean_squared_error(y_test, y_pred))
print('R2 score: ', r2_score(y_test, y_pred))
print('Index of agreement: ', ia)
