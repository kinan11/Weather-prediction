import pandas as pd


# dane do testowania w 2022
def get_xy1(n):
    data = pd.read_csv('Dataset_to_predict.csv')
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
