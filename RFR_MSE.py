from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    n = 25
    for i in range(n):
        x, y = get_xy(i+1)

        feature_names.extend(
            ["T_max_" + str(i + 1), "T_min_" + str(i + 1), "T_śr_" + str(i + 1), "Suma opadów_" + str(i + 1),
             "Śr. wilgotność_" + str(i + 1), "Śr. pr. wiatru_" + str(i + 1), "Śr. zachmurzenie_" + str(i + 1),
             "Ciśnienie_" + str(i + 1)])

        # Podział na dane uczące i trenujące
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        st_x = StandardScaler()
        x_train = st_x.fit_transform(x_train)
        x_test = st_x.transform(x_test)

        regressor_RFR = RandomForestRegressor(max_depth=100, random_state=0, n_estimators=200)
        regressor_RFR.fit(x_train, y_train)
        y_pred_LR = regressor_RFR.predict(x_test)

        ia.append(1 - (np.sum((y_test - y_pred_LR) ** 2)) / (
            np.sum((np.abs(y_pred_LR - np.mean(y_test)) + np.abs(y_test - np.mean(y_test))) ** 2)))
        mse.append(mean_squared_error(y_test, y_pred_LR))
        r2score.append(r2_score(y_test, y_pred_LR))

    fig, axs = plt.subplots(3, 1, constrained_layout=True)

    axs[0].plot([i + 1 for i in range(n)], mse, color="red")
    axs[0].set_title('MSE')
    axs[0].set_xlabel('days')
    axs[0].set_ylabel("mse [$^\circ$$C^{2}$]")

    axs[1].plot([i + 1 for i in range(n)], r2score, color="blue")
    axs[1].set_title('R2 score')
    axs[1].set_xlabel('days')
    axs[1].set_ylabel('R2 score [%]')

    axs[2].plot([i + 1 for i in range(n)], ia, color="green")
    axs[2].set_title('Index of agreement')
    axs[2].set_xlabel('days')
    axs[2].set_ylabel('Index of agreement [%]')

    plt.savefig('./Wykresy/RFR_MSE'+str(n)+'.png', format='png')

    plt.show()

    feature_names.reverse()

    explainer = shap.Explainer(regressor_RFR.predict, x_test)
    shap_values = explainer(x_test)
    shap.summary_plot(shap_values, show=False, feature_names=feature_names, plot_type="bar")
    plt.savefig('./Wykresy/RFR_'+str(n)+'days.png', format='png')
    plt.close()


if __name__ == '__main__':
    main()
