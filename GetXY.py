import pandas as pd


# dane standardowe, ze wszystkimi wartościami
def get_xy(n):
    data = pd.read_csv('Dataset.csv')
    data = data.drop('Dzień', axis=1)
    data = data.drop('Miesiąc', axis=1)
    data = data.drop('Rok', axis=1)

    # Pozyskanie danych
    x_tmp = data.iloc[:, 1:].values
    y = data.iloc[n:, 3].values

    x = []
    for i in range(len(x_tmp) - n):
        x.append([])
        for j in range(n):
            x[i].extend(x_tmp[i+j])
    return x, y


# dane tylko z temperaturą (temperatura min, max, średnia)
def get_xy2(n):
    data = pd.read_csv('Dataset.csv')
    data = data.drop('Dzień', axis=1)
    data = data.drop('Miesiąc', axis=1)
    data = data.drop('Rok', axis=1)
    data = data.drop('Suma dobowa opadów', axis=1)
    data = data.drop('Średnia wilgotność', axis=1)
    data = data.drop('Średnia prędkość wiatru', axis=1)
    data = data.drop('Średnie zachmurzenie', axis=1)
    data = data.drop('Ciśnienie na stacji', axis=1)

    # Pozyskanie danych
    x_tmp = data.iloc[:, 1:].values
    y = data.iloc[n:, 3].values

    x = []
    for i in range(len(x_tmp) - n):
        x.append([])
        for j in range(n):
            x[i].extend(x_tmp[i+j])
    return x, y

