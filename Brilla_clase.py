import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class BrillaClassifier(object):
    def __init__(self, files: list, numericas: list, categoricas: list, tamano_entrenamiento=0.9):
        self.files = files
        self.df = self.construccion_base_de_datos()
        self.correccion_df()
        self.x = self.df[numericas + categoricas]
        self.y = self.df['categoria']
        self.numericas = numericas
        self.categoricas = categoricas
        self.tamano_entrenamiento = tamano_entrenamiento
        self.x_train, self.x_test, self.y_train, self.y_test = self.separacion_train_test()

    def construccion_base_de_datos(self):
        """
        DOCUMENTACION
        :return:
        """
        num_files = range(len(self.files))
        for file, index in zip(self.files, num_files):
            file['categoria'] = [index] * len(file)

        df_final = pd.concat(self.files, axis=0, ignore_index=True)
        return shuffle(df_final).reset_index(drop=True)

    def correccion_df(self):
        """
        DOCUMENTACION
        :return:
        """
        self.df['FECHA_NACIMIENTO'] = pd.to_datetime(self.df['FECHA_NACIMIENTO'], format='%m/%d/%Y')
        self.df['EDAD'] = (pd.Timestamp('now') - self.df['FECHA_NACIMIENTO']).astype('<m8[Y]')

        self.df['CUPO_DISPONIBLE'] = self.df['CUPO_DISPONIBLE'].str.replace('$', '', regex=True)
        self.df['CUPO_DISPONIBLE'] = (self.df['CUPO_DISPONIBLE'].str.replace(',', '', regex=True)).astype(int)

        self.df['ESTRATO'] = self.df['ESTRATO'].astype(str)
        self.df['SEGMENTO'] = np.where(self.df['SEGMENTO'].isnull(), 'VACIO', self.df.SEGMENTO)

    def separacion_train_test(self):
        """
        DOCUMENTACION
        :return:
        """
        self.column_transformer = ColumnTransformer([('ohe', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), self.categoricas),
                                                ('scaler', MinMaxScaler(), self.numericas)])
        # now we can pass the full dataset, as the column transformer will do the subsetting for us:
        self.column_transformer.fit(self.x)
        x = self.column_transformer.transform(self.x)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(self.y.values)
        return train_test_split(x, y, random_state=1, test_size=1 - self.tamano_entrenamiento)

    def evaluar_clasificador(self, clasificador):
        """
        DOCUMENTACION
        :param matriz_confusion:
        :param clasificador:
        :return:
        """
        print('Accuracy  on training set: {:.2f}'.format(clasificador.score(self.x_train, self.y_train)))
        print('Accuracy of K-NN classifier on test set: {:.2f}'.format(clasificador.score(self.x_test, self.y_test)))

        predicciones = clasificador.predict(self.x_test)
        matrix=confusion_matrix(self.y_test, predicciones)
        print(classification_report(self.y_test, predicciones))
        return predicciones, matrix
    

    def clasificar_KNN(self, num_vecinos=5, metrica='minkowski', p=1, pesos='distance'):
        """
        DOCUMENTACION
        :param pesos:
        :param p:
        :param metrica:
        :param num_vecinos:
        :return:
        """
        clasificador = KNeighborsClassifier(n_neighbors=num_vecinos, metric=metrica, p=p, weights=pesos)
        clasificador.fit(self.x_train, self.y_train)
        predicciones, self.matrix = self.evaluar_clasificador(clasificador=clasificador)
        return predicciones, clasificador

    def clasificar_random_forest(self, n_estimadores=25, criterio='entropy'):
        """
        DOCUMENTACION
        :return:
        """
        clasificador = RandomForestClassifier(n_estimators=n_estimadores, criterion=criterio, random_state=0)
        clasificador.fit(self.x_train, self.y_train)
        predicciones, self.matrix = self.evaluar_clasificador(clasificador=clasificador)
        return predicciones, clasificador

    def clasificar_ADA_boost(self):
        """
        DOCUMENTACION
        :return:
        """
        clasificador = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=300)
        clasificador.fit(self.x_train, self.y_train)
        predicciones, self.matrix = self.evaluar_clasificador(clasificador=clasificador)
        return predicciones, clasificador


def main_brilla(files: list, numericas: list, categoricas: list, clasificador: str, tamano_entrenamiento=0.9, 
                matriz_confusion=False, df_gases=None):
    brilla_clasificador = BrillaClassifier(files=files,
                                           numericas=numericas,
                                           categoricas=categoricas,
                                           tamano_entrenamiento=tamano_entrenamiento)

    if clasificador == 'knn':
        predicciones, clasificador = brilla_clasificador.clasificar_KNN()
    elif clasificador == 'random_forest':
        predicciones, clasificador = brilla_clasificador.clasificar_random_forest()
    else:
        predicciones, clasificador = brilla_clasificador.clasificar_ADA_boost()
    if df_gases is not None:
        X_gases = brilla_clasificador.column_transformer.transform(df_gases)
        predicciones_df_gases = clasificador.predict(X_gases)
    else: predicciones_df_gases = []
        
    if matriz_confusion: return predicciones, clasificador, brilla_clasificador.matrix
    else: return predicciones, predicciones_df_gases


if __name__ == '__main__':
    import os
    directory = os.getcwd()
    path =  directory + "\data"

    df_electrodomesticos = pd.read_csv(path + '\BD_Brilla_elctrodomesticos.csv', on_bad_lines='skip')
    df_construccion = pd.read_csv(path + '\BD_Brilla_contruccion.csv', on_bad_lines='skip')
    df_motos = pd.read_csv(path + '\BD_Brilla_motos.csv', on_bad_lines='skip')
    df_tecnologia = pd.read_csv(path + '\BD_Brilla_tecnologia.csv', on_bad_lines='skip')
    df_control = pd.read_csv(path + '\BD_Brilla_control.csv', on_bad_lines='skip')

    archivos = [df_electrodomesticos, df_control, df_construccion, df_motos, df_tecnologia]

    numeric = ['CUPO_DISPONIBLE', 'ESTRATO', 'EDAD']
    categorical = ['GENERO', 'SEGMENTO', 'BARRIO', 'LOCALIDAD', 'DEPARTAMENTO']

    print('Con todas las bases de datos')
    pred1, _ = main_brilla(files=archivos,
                        numericas=numeric,
                        categoricas=categorical,
                        clasificador='random_forest')

    print('Sin Electrodomesticos ni tecnología')
    archivos2 = [df_control, df_construccion, df_motos]
    pred_no_tec_elec, _ = main_brilla(files=archivos2,
                                   numericas=numeric,
                                   categoricas=categorical,
                                   clasificador='random_forest')

    print('Sin Electrodomesticos')
    archivos3 = [df_control, df_construccion, df_motos, df_tecnologia]
    pred_no_elec, _ = main_brilla(files=archivos3,
                                   numericas=numeric,
                                   categoricas=categorical,
                                   clasificador='random_forest')

    print('Sin Tecnologia')
    archivos4 = [df_control, df_construccion, df_motos, df_electrodomesticos]
    pred_no_tec, _ = main_brilla(files=archivos4,
                               numericas=numeric,
                               categoricas=categorical,
                               clasificador='random_forest')
