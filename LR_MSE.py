from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
from GetXY import get_xy
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap


def main():
    mse = []
    r2score = []
    ia = []
    feature_names = []
    n = 10
    for i in range(n):
        x, y = get_xy(i+1)

        feature_names.extend(
            ["T_max_" + str(n - i), "T_min_" + str(n - i), "T_śr_" + str(n - i), "Suma opadów_" + str(n - i),
             "Śr. wilgotność_" + str(n - i), "Śr. pr. wiatru_" + str(n - i), "Śr. zachmurzenie_" + str(n - i),
             "Ciśnienie_" + str(n - i)])

        # Podział na dane uczące i trenujące
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        st_x = StandardScaler()
        x_train = st_x.fit_transform(x_train)
        x_test = st_x.transform(x_test)

        regressor_LR = LinearRegression()
        regressor_LR.fit(x_train, y_train)
        y_pred_LR = regressor_LR.predict(x_test)

        ia.append(1 - (np.sum((y_test - y_pred_LR) ** 2)) / (
            np.sum((np.abs(y_pred_LR - np.mean(y_test)) + np.abs(y_test - np.mean(y_test))) ** 2)))
        mse.append(mean_squared_error(y_test, y_pred_LR))
        r2score.append(r2_score(y_test, y_pred_LR))

    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    axs[0].plot([i + 1 for i in range(n)], mse, color="red")
    axs[0].set_title('MSE')
    axs[0].set_xlabel('days')
    axs[0].set_ylabel("mse [$^\circ$$C^{2}$]")
    axs[0].text(0.7, max(mse) - 0.2, 'min: ' + str(round(min(mse), 2)), bbox=props)

    axs[1].plot([i + 1 for i in range(n)], r2score, color="blue")
    axs[1].set_title('R2 score')
    axs[1].set_xlabel('days')
    axs[1].set_ylabel('R2 score')
    axs[1].text(0.7, max(r2score) - 0.003, 'max: ' + str(round(max(r2score), 4)), bbox=props)

    axs[2].plot([i + 1 for i in range(n)], ia, color="green")
    axs[2].set_title('Index of agreement')
    axs[2].set_xlabel('days')
    axs[2].set_ylabel('Index of agreement')
    axs[2].text(0.7, max(ia) - 0.001, 'max: ' + str(round(max(ia), 4)), bbox=props)

    plt.savefig('./Wykresy/LR_MSE'+str(n)+'.png', format='png')

    plt.show()

    explainer = shap.Explainer(regressor_LR.predict, x_test)
    shap_values = explainer(x_test)
    shap.summary_plot(shap_values, show=False, feature_names=feature_names, plot_type="bar")
    plt.savefig('./Wykresy/LR_'+str(n)+'days.png', format='png')
    plt.close()


if __name__ == '__main__':
    main()
