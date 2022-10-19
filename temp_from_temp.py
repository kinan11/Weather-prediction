import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main():
    data = pd.read_csv('Dataset.csv')

    # Pozyskanie danych
    x_tmp = data.iloc[:, 6].values
    y = data.iloc[5:, 6].values

    x = []
    for i in range(len(x_tmp)-5):
        x.append([x_tmp[i], x_tmp[i+1], x_tmp[i+2], x_tmp[i+3], x_tmp[i+4]])

    # Podział na dane uczące i trenujące
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Regresja liniowa
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # Wyniki
    print('Train Score: ', regressor.score(x_train, y_train))
    print('Test Score: ', regressor.score(x_test, y_test))


if __name__ == '__main__':
    main()
