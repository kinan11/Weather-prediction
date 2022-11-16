import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

from GetXY import get_xy


def build_model():
    model = Sequential()
    model.add(Dense(19, input_dim=80, activation='sigmoid'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


n = 10

build_model().summary()

feature_names = []

x, y = get_xy(n)

for i in range(n):
    x, y = get_xy(i + 1)

    feature_names.extend(
        ["T_max_" + str(n - i), "T_min_" + str(n - i), "T_śr_" + str(n - i), "Suma opadów_" + str(n - i),
         "Śr. wilgotność_" + str(n - i), "Śr. pr. wiatru_" + str(n - i), "Śr. zachmurzenie_" + str(n - i),
         "Ciśnienie_" + str(n - i)])

# data = pd.read_csv('Dataset.csv')
# data = data.drop('Dzień', axis=1)
# data = data.drop('Miesiąc', axis=1)
# data = data.drop('Rok', axis=1)
#
# # Pozyskanie danych
# x_tmp = data.iloc[:, 1:].values
# y = data.iloc[4:, 3].values
#
# x = []
# for i in range(len(x_tmp) - 4):
#     x.append([])
#     x[i].extend(x_tmp[i])
#     x[i].extend(x_tmp[i + 1])
#     x[i].extend(x_tmp[i + 2])
#     x[i].extend(x_tmp[i + 3])

# Podział na dane uczące i trenujące
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Skalowanie danych
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

build_model = build_model()

history = build_model.fit(x_train, y_train, epochs=50, verbose=0, validation_split=0.2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, 10])
plt.xlabel('Epoch')
plt.ylabel('Error [MPG]')
plt.legend()
plt.grid(True)
plt.show()
print(history.history['val_loss'].index(min(history.history['val_loss'])))

# y_pred = build_model.predict(x_test)
# ia = (1 - (np.sum((y_test - y_pred) ** 2)) /
#       (np.sum((np.abs(y_pred - np.mean(y_test)) + np.abs(y_test - np.mean(y_test))) ** 2)))
# print(str(y_pred[0]) + " " + str(y_test[0]))

explainer = shap.Explainer(build_model.predict, x_test)
shap_values = explainer(x_test)
shap.summary_plot(shap_values, show=False, feature_names=feature_names, plot_type="bar")
plt.savefig('./Wykresy/NN_'+str(n)+'days.png', format='png')
plt.close()

# print('Neural Network: ')
# print('Mean squared error: ', mean_squared_error(y_test, y_pred))
# print('R2 score: ', r2_score(y_test, y_pred))
# print('Index of agreement: ', ia)
