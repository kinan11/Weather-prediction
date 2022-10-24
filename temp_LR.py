import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor


def main():
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
        x[i].extend(x_tmp[i+1])
        x[i].extend(x_tmp[i + 2])
        x[i].extend(x_tmp[i + 3])

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


if __name__ == '__main__':
    main()

# Sieć neuronowa z tutoriala
