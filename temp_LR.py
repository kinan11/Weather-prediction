from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
import shap
from GetXY import get_xy
from GetXY_predict import get_xy1


def main():
    # pozyskanie danych z 3 poprzednich dni
    x, y = get_xy(3)

    feature_names = []

    # nagłówki do analizy shap
    for i in range(3):
        feature_names.extend(["T_max_"+str(i+1), "T_min_"+str(i+1), "T_śr_"+str(i+1), "Suma opadów_"+str(i+1),
                            "Śr. wilgotność_"+str(i+1), "Śr. pr. wiatru_"+str(i+1), "Śr. zachmurzenie_"+str(i+1),
                            "Ciśnienie_"+str(i+1)])

    # Podział na dane uczące i trenujące
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Skalowanie danych
    st_x = StandardScaler()
    x_train = st_x.fit_transform(x_train)
    x_test = st_x.transform(x_test)

    # Algorytmy
    regressor_LR = LinearRegression()
    regressor_DTR = DecisionTreeRegressor()
    regressor_RFR = RandomForestRegressor(max_depth=100, random_state=0, n_estimators=200)
    regressor_ETR = ExtraTreeRegressor()

    # Dopasowanie
    regressor_LR.fit(x_train, y_train)
    regressor_DTR.fit(x_train, y_train)
    regressor_RFR.fit(x_train, y_train)
    regressor_ETR.fit(x_train, y_train)

    # Predykcja
    y_pred_LR = regressor_LR.predict(x_test)
    y_pred_DTR = regressor_DTR.predict(x_test)
    y_pred_RFR = regressor_RFR.predict(x_test)
    y_pred_ETR = regressor_ETR.predict(x_test)

    # analiza shap
    # explainer = shap.Explainer(regressor_LR.predict, x_test)
    # shap_values = explainer(x_test)
    # shap.summary_plot(shap_values, show=False, feature_names=feature_names, plot_type="bar")
    # plt.savefig('./Wykresy/LR_10days.png', format='png')
    # plt.close()

    # explainer = shap.Explainer(regressor_DTR.predict, x_test)
    # shap_values = explainer(x_test)
    # shap.plots.bar(shap_values)

    # Index od agreement
    ia_LR = (1 - (np.sum((y_test - y_pred_LR) ** 2)) / (
        np.sum((np.abs(y_pred_LR - np.mean(y_test)) + np.abs(y_test - np.mean(y_test))) ** 2)))
    ia_DTR = (1 - (np.sum((y_test - y_pred_DTR) ** 2)) / (
        np.sum((np.abs(y_pred_DTR - np.mean(y_test)) + np.abs(y_test - np.mean(y_test))) ** 2)))
    ia_RFR = (1 - (np.sum((y_test - y_pred_RFR) ** 2)) / (
        np.sum((np.abs(y_pred_RFR - np.mean(y_test)) + np.abs(y_test - np.mean(y_test))) ** 2)))
    ia_ETR = (1 - (np.sum((y_test - y_pred_ETR) ** 2)) / (
        np.sum((np.abs(y_pred_ETR - np.mean(y_test)) + np.abs(y_test - np.mean(y_test))) ** 2)))

    # Wyniki
    print('Linear Regression: ')
    print('Mean squared error: ', mean_squared_error(y_test, y_pred_LR))
    print('R2 score: ', r2_score(y_test, y_pred_LR))
    print('Index of agreement: ', ia_LR)

    print('\nDecision Tree Regression:')
    print('Mean squared error: ', mean_squared_error(y_test, y_pred_DTR))
    print('R2 score: ', r2_score(y_test, y_pred_DTR))
    print('Index of agreement: ', ia_DTR)

    print('\nRandom Forest Regression:')
    print('Mean squared error: ', mean_squared_error(y_test, y_pred_RFR))
    print('R2 score: ', r2_score(y_test, y_pred_RFR))
    print('Index of agreement: ', ia_RFR)

    print('\nExtra Tree Regression:')
    print('Mean squared error: ', mean_squared_error(y_test, y_pred_ETR))
    print('R2 score: ', r2_score(y_test, y_pred_ETR))
    print('Index of agreement: ', ia_ETR)

    # pozyskanie danych z poprzednich 3 dni dla 2022
    x, y = get_xy1(3)
    st_x = StandardScaler()
    x = st_x.fit_transform(x)
    y_pred = regressor_LR.predict(x)
    ia = (1 - (np.sum((y - y_pred) ** 2)) /
          (np.sum((np.abs(y_pred - np.mean(y)) + np.abs(y - np.mean(y))) ** 2)))

    print('\nLinear Regression for 2022: ')
    print('Mean squared error: ', mean_squared_error(y, y_pred))
    print('R2 score: ', r2_score(y, y_pred))
    print('Index of agreement: ', ia)
    plt.plot(y_pred, color="red")
    plt.plot(y, color="blue")
    plt.savefig('./Wykresy/LR_2022.png', format='png')
    plt.show()


if __name__ == '__main__':
    main()
