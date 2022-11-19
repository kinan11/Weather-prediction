import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from GetXY_predict import get_xy1
from tensorflow import keras


def main():
    n = 3
    x, y = get_xy1(n)
    feature_names = []

    for i in range(n):
        feature_names.extend(
            ["T_max_" + str(n - i), "T_min_" + str(n - i), "T_śr_" + str(n - i), "Suma opadów_" + str(n - i),
             "Śr. wilgotność_" + str(n - i), "Śr. pr. wiatru_" + str(n - i), "Śr. zachmurzenie_" + str(n - i),
             "Ciśnienie_" + str(n - i)])

    st_x = StandardScaler()
    x = st_x.fit_transform(x)

    model = keras.models.load_model('ssn_model')
    model_LR = pickle.load(open('model_LN', 'rb'))

    y_pred = model.predict(x)
    ia = (1 - (np.sum((y - y_pred) ** 2)) /
          (np.sum((np.abs(y_pred - np.mean(y)) + np.abs(y - np.mean(y))) ** 2)))

    print('Neural Network: ')
    print('Mean squared error: ', mean_squared_error(y, y_pred))
    print('R2 score: ', r2_score(y, y_pred))
    print('Index of agreement: ', ia)

    plt.plot(y_pred, color="red")
    plt.plot(y, color="blue")
    plt.savefig('./Wykresy/NN_2022.png', format='png')
    # plt.set_title('MSE')
    # plt.set_xlabel('days')
    # plt.set_ylabel("mse [$^\circ$$C^{2}$]")
    plt.show()

    # y_pred = model_LR.predict(x)
    # ia = (1 - (np.sum((y - y_pred) ** 2)) /
    #       (np.sum((np.abs(y_pred - np.mean(y)) + np.abs(y - np.mean(y))) ** 2)))
    #
    # print('Neural Network: ')
    # print('Mean squared error: ', mean_squared_error(y, y_pred))
    # print('R2 score: ', r2_score(y, y_pred))
    # print('Index of agreement: ', ia)

    # plt.plot(y_pred, color="red")
    # plt.plot(y, color="blue")
    # # plt.set_title('MSE')
    # # plt.set_xlabel('days')
    # # plt.set_ylabel("mse [$^\circ$$C^{2}$]")
    # plt.show()
    # score = model_LR.score(x, y)
    # print(score)


if __name__ == '__main__':
    main()
