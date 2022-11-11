import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap


def main():
    data = pd.read_csv('Dataset.csv')

    # Pozyskanie danych
    x_tmp = data.iloc[:, 6].values
    mse = []
    r2score = []
    ia = []
    regressor = LinearRegression()
    x_test = []

    for i in range(10):
        y = data.iloc[i+1:, 6].values

        x = []
        for j in range(len(x_tmp) - (i+1)):
            x.append([])
            for k in range(i+1):
                x[j].append(x_tmp[j+k])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        # Regresja liniowa
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)

        y_pred = regressor.predict(x_test)

        mse.append(mean_squared_error(y_test, y_pred))
        r2score.append(r2_score(y_test, y_pred))

        ia.append(1 - (np.sum((y_test - y_pred) ** 2)) / (
            np.sum((np.abs(y_pred - np.mean(y_test)) + np.abs(y_test - np.mean(y_test))) ** 2)))

    fig, axs = plt.subplots(3, 1, constrained_layout=True)

    axs[0].plot([i + 1 for i in range(10)], mse, color="red")
    axs[0].set_title('MSE')
    axs[0].set_xlabel('days')
    axs[0].set_ylabel("mse [$^\circ$$C^{2}$]")

    axs[1].plot([i + 1 for i in range(10)], r2score, color="blue")
    axs[1].set_title('R2 score')
    axs[1].set_xlabel('days')
    axs[1].set_ylabel('R2 score [%]')

    axs[2].plot([i + 1 for i in range(10)], ia, color="green")
    axs[2].set_title('Index of agreement')
    axs[2].set_xlabel('days')
    axs[2].set_ylabel('Index of agreement [%]')

    plt.savefig('./Temp_from_temp/MSE200.png', format='png')
    plt.show()

    x_test = np.array(x_test)

    explainer = shap.Explainer(regressor.predict, x_test)
    shap_values = explainer(x_test)
    feature_names = [str(10-i) for i in range(10)]
    shap.summary_plot(shap_values, show=False, feature_names=feature_names, plot_type="bar")
    plt.ylabel('Days',fontsize=15, labelpad=1)
    plt.savefig('./Temp_from_temp/10days.png', format='png')
    plt.close()


if __name__ == '__main__':
    main()
