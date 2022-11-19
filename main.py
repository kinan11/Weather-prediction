import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    data = pd.read_csv('Dataset.csv')

    # Pozyskanie danych
    x = data.iloc[:-1, 1:].values
    y = data.iloc[1:, 6].values

    # Podział na dane uczące i trenujące
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Skalowanie danych
    st_x = StandardScaler()
    x_train = st_x.fit_transform(x_train)
    x_test = st_x.transform(x_test)

    print(x_test)
    print(x_train)


if __name__ == '__main__':
    main()

