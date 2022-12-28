import numpy
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import shap


def main():
    data = pd.read_csv('Dataset.csv')

    # Pozyskanie danych
    x_tmp = data.iloc[:, 6].values
    y = data.iloc[3:, 6].values

    x = []
    for i in range(len(x_tmp)-3):
        x.append([x_tmp[i], x_tmp[i+1], x_tmp[i+2]])


    # Podział na dane uczące i trenujące
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Regresja liniowa
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_test)

    x_test = numpy.array(x_test)

    explainer = shap.Explainer(regressor.predict, x_test)
    shap_values = explainer(x_test)
    feature_names = ["3","2","1"]
    shap.summary_plot(shap_values,show=False,feature_names=feature_names, plot_type="bar")
    plt.savefig('./Temp_from_temp/3days.png', format='png')
    plt.close()

    ia = (1 - (np.sum((y_test - y_pred) ** 2)) / (
        np.sum((np.abs(y_pred - np.mean(y_test)) + np.abs(y_test - np.mean(y_test))) ** 2)))

    print('Mean squared error: ', mean_squared_error(y_test, y_pred))
    print('R2 score: ', r2_score(y_test, y_pred))
    print('Index of agreement: ', ia)

    # Wyniki
    print('\nTrain Score: ', regressor.score(x_train, y_train))
    print('Test Score: ', regressor.score(x_test, y_test))


if __name__ == '__main__':
    main()
