from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import tensorflow as tf
from GetXY import get_xy


def build_model(activation1='sigmoid', activation2='relu', activation3='linear', loss="mean_squared_error",
                optimizer="adam"):
    model = Sequential()
    model.add(Dense(11, input_dim=24, activation=activation1))
    model.add(Dense(4, activation=activation2))
    model.add(Dense(1, activation=activation3))
    model.compile(loss=loss, optimizer=optimizer)
    return model


n = 3

build_model().summary()

feature_names = []

x, y = get_xy(n)

# Podział na dane uczące i trenujące
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Skalowanie danych
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

# hiper parametry do wyboru
param_grid = {
    'activation1': ['relu', 'sigmoid', 'linear'],
    'activation2': ['relu', 'sigmoid', 'linear'],
    'activation3': ['relu', 'sigmoid', 'linear'],
    'optimizer': ['adam', 'rmsprop'],
    'loss': ['mse', 'mae'],
}

# warunek wcześniejszego zakończenia
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=10)

# grid serach
grid_search = GridSearchCV(
    estimator=KerasRegressor(build_model, verbose=0, validation_split=0.2, callbacks=[callback], epochs=300),
    param_grid=param_grid,
)

grid_search.fit(x_train, y_train, verbose=0)

y_pred = grid_search.predict(x_test)
ia = (1 - (np.sum((y_test - y_pred) ** 2)) /
      (np.sum((np.abs(y_pred - np.mean(y_test)) + np.abs(y_test - np.mean(y_test))) ** 2)))


print('Neural Network with Grid Search: ')
print('Mean squared error: ', mean_squared_error(y_test, y_pred))
print('R2 score: ', r2_score(y_test, y_pred))
print('Index of agreement: ', ia)
print('Best params: ', grid_search.best_params_)

# build_model.save('ssn_model_grid')
