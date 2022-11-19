import pandas as pd
import os


def k_d():
    col_list = ['Kod stacji', 'Nazwa stacji', 'Rok', 'Miesiąc', 'Dzień', 'T_MAX', 'Status', 'T_MIN', 'Status', 'T_ŚR',
                'Status', 'T_MIN w gruncie', 'Status', 'Suma dobowa opadów', 'Status', 'Rodzaj opadów',
                'Wysokość pokrywy śnieżnej', 'Status']

    df = pd.DataFrame({'Rok': [], 'Miesiąc': [], 'Dzień': [], 'T_MAX': [], 'T_MIN': [], 'T_ŚR': [],
                       'Suma dobowa opadów': []})

    for f in os.listdir('Data_to_predict'):
        if f.startswith('k_d_0') or f.startswith('k_d_1'):
            data = pd.read_csv("Data_to_predict/"+f, encoding="ISO-8859-1", header=None)
            data.columns = col_list

            for index, row in data.iterrows():
                if row['Nazwa stacji'] == "KRAKÓW-OBSERWATORIUM":
                    r = pd.DataFrame({'Rok': [row['Rok']], 'Miesiąc': [row['Miesiąc']],
                                      'Dzień': [row['Dzień']], 'T_MAX': [row['T_MAX']],
                                      'T_MIN': [row['T_MIN']], 'T_ŚR': [row['T_ŚR']],
                                      'Suma dobowa opadów': [row['Suma dobowa opadów']]})

                    df = pd.concat([df, r])

    df = df.reset_index(drop=True)
    return df


def k_t():
    col_list = ['Kod stacji', 'Nazwa stacji', 'Rok', 'Miesiąc', 'Dzień', 'T_ŚR', 'Status', 'Średnia wilgotność',
                'Status', 'Średnia prędkość wiatru', 'Status', 'Średnie zachmurzenie', 'Status']
    df = pd.DataFrame({'Rok': [], 'Miesiąc': [], 'Dzień': [], 'Średnia wilgotność': [], 'Średnia prędkość wiatru': [],
                       'Średnie zachmurzenie': []})

    for f in os.listdir('Data_to_predict'):
        if f.startswith('k_d_t'):
            data = pd.read_csv("Data_to_predict/" + f, encoding="ISO-8859-1", header=None)
            data.columns = col_list

            for index, row in data.iterrows():
                if row['Nazwa stacji'] == "KRAKÓW-OBSERWATORIUM":
                    r = pd.DataFrame({'Rok': [row['Rok']], 'Miesiąc': [row['Miesiąc']],
                                      'Dzień': [row['Dzień']], 'Średnia wilgotność': [row['Średnia wilgotność']],
                                      'Średnia prędkość wiatru': [row['Średnia prędkość wiatru']],
                                      'Średnie zachmurzenie': [row['Średnie zachmurzenie']]})

                    df = pd.concat([df, r])

    df = df.reset_index(drop=True)
    return df


def s_d_t():
    col_list = ['Kod stacji', 'Nazwa stacji', 'Rok', 'Miesiąc', 'Dzień', 'Średnie zachmurzenie', 'Status',
                'Średnia prędkość wiatru', 'Status', 'Średnia temperatura', 'Status', 'Średnie ciśnienie pary wodnej',
                'Status', 'Średnia wilgotność', 'Status', 'Ciśnienie na stacji', 'Status', 'Ciśnienie nad morzem',
                'Status', 'Suma opadu dzień', 'Status', 'Suma opadu noc', 'Status']
    df = pd.DataFrame({'Rok': [], 'Miesiąc': [], 'Dzień': [], 'Ciśnienie na stacji': []})

    for f in os.listdir('Data_to_predict'):
        if f.startswith('s'):
            data = pd.read_csv("Data_to_predict/" + f, encoding="ISO-8859-1", header=None)
            data.columns = col_list

            for index, row in data.iterrows():
                if row['Nazwa stacji'] == "KRAKÓW-BALICE":
                    r = pd.DataFrame({'Rok': [row['Rok']], 'Miesiąc': [row['Miesiąc']],
                                      'Dzień': [row['Dzień']], 'Ciśnienie na stacji': [row['Ciśnienie na stacji']]})
                    df = pd.concat([df, r])

    df = df.reset_index(drop=True)
    return df


def main():

    df_d = k_d()
    df_t = k_t()
    df_s = s_d_t()

    df = pd.merge(df_d, df_t)
    df = pd.merge(df, df_s)

    df.to_csv('Dataset_to_predict.csv')


if __name__ == '__main__':
    main()
