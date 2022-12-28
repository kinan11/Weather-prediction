import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from GetXY import get_xy


def main():
    # pozyskanie danych z 3 poprzednich dni
    x, y = get_xy(3)

    # Podział na dane uczące i trenujące
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Skalowanie danych
    st_x = StandardScaler()
    x_train = st_x.fit_transform(x_train)
    x_test = st_x.transform(x_test)

    # najlepsze mse
    mm = 10
    # najlepsza liczba sąsiadów
    mi = 0

    # poszukiwanie najlepszej liczby sąsiadów
    for i in range(50):
        regressor = neighbors.KNeighborsRegressor(n_neighbors=(i+1))
        regressor.fit(x_train, y_train)

        y = regressor.predict(x_test)

        ia = (1 - (np.sum((y_test - y) ** 2)) / (
            np.sum((np.abs(y - np.mean(y_test)) + np.abs(y_test - np.mean(y_test))) ** 2)))

        if mean_squared_error(y_test, y) < mm:
            mm = mean_squared_error(y_test, y)
            mi = i

        print('\n' + str(i) +' neighbours: ')
        print('Mean squared error: ', mean_squared_error(y_test, y))
        print('R2 score: ', r2_score(y_test, y))
        print('Index of agreement: ', ia)

    regressor = neighbors.KNeighborsRegressor(n_neighbors=(mi + 1))
    regressor.fit(x_train, y_train)

    print("\n Best neighbours: " + str(mi) + "\n MSE: "+str(mm) )


if __name__ == '__main__':
    main()
