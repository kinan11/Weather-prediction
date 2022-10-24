import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


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




if __name__ == '__main__':
    main()
