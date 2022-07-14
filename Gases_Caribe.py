import pandas as pd
import os

class GasCleaner(object):
    def __init__(self, path: str):
        self.Gases = pd.read_excel(path)
        
    def cleaner_df(self):
        df =  self.Gases[['CUPO_DISPONIBLE', 'SUBCATEGORIA', 'FECHA_NACIMIENTO', 'GENERO', 'DESC_SEGMENTACION', 'DESC_BARRIO', 'DESC_LOCALIDAD', 'DESC_DPTO']]
        df = df.rename(columns = {'DESC_DPTO': 'DEPARTAMENTO', 'DESC_LOCALIDAD': 'LOCALIDAD', 'DESC_BARRIO': 'BARRIO', 'SUBCATEGORIA': 'ESTRATO', 'DESC_SEGMENTACION': 'SEGMENTACION'})
        df['ESTRATO'] = df['ESTRATO'].str.replace('ESTRATO ', '').astype(int)
        #df['CUPO_DISPONIBLE'] = df['CUPO_DISPONIBLE'].str.replace(',', '').astype(float)
        #df['DIRECCION'] = df['DIRECCION'].str.split('-', expand=True)[0]
        no_age_index = df.sort_values(by='FECHA_NACIMIENTO', ascending=False)[:4].index
        df.drop(no_age_index, axis=0, inplace=True)
        df = df.dropna(axis=0)
        df['FECHA_NACIMIENTO'] = pd.to_datetime(df['FECHA_NACIMIENTO'], format='%Y-%m-%d')
        df['EDAD']=(pd.Timestamp('now') - df['FECHA_NACIMIENTO']).astype('<m8[Y]')
        too_old_index = df[df['EDAD'] > 95].index
        df.drop(too_old_index, axis=0, inplace=True)

        clasificar_genero = lambda x: 'Femenino' if x=='F' else 'Masculino'
        df['GENERO'] = df['GENERO'].apply(clasificar_genero)

        df = df[['CUPO_DISPONIBLE', 'DEPARTAMENTO', 'LOCALIDAD', 'BARRIO', 'ESTRATO', 'GENERO', 'SEGMENTACION', 'EDAD']]

        df_segmentacion = df['SEGMENTACION']
        df_segmentacion = df_segmentacion.str.split(' - ', expand=True)
        df['SEGMENTO'] = df_segmentacion[1]
        self.df_gasesC = df[['CUPO_DISPONIBLE', 'DEPARTAMENTO', 'LOCALIDAD', 'BARRIO', 'ESTRATO', 'GENERO', 'SEGMENTO', 'EDAD']]
        return self.df_gasesC


    def grafico_numerico(self, variable):
        self.df_gasesC.hist(column=variable, bins=80, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
        
    def grafico_categorico(self, variable):
        self.df_gasesC[variable].value_counts().plot(ArithmeticErrorkind='bar').set_title(variable)
        
    
if __name__ == '__main__':
    path = os.getcwd()
    Gas_cleaner = GasCleaner(path + '\data\BD_GasesDelCaribe_Potenciales.xlsx')
    df_gases_caribe = Gas_cleaner.cleaner_df()
# #%%   
#     Gas_cleaner.grafico_numerico('EDAD')
#     Gas_cleaner.grafico_numerico('ESTRATO')
# #%%    
#     Gas_cleaner.grafico_categorico('DEPARTAMENTO')
# #%% 
#     Gas_cleaner.grafico_categorico('GENERO')
# #%% 
#     Gas_cleaner.grafico_categorico('SEGMENTO')

    df_electrodomesticos = pd.read_csv(path + '\data\BD_Brilla_elctrodomesticos.csv', on_bad_lines='skip')
    df_construccion = pd.read_csv(path + '\data\BD_Brilla_contruccion.csv', on_bad_lines='skip')
    df_motos = pd.read_csv(path + '\data\BD_Brilla_motos.csv', on_bad_lines='skip')
    df_tecnologia = pd.read_csv(path + '\data\BD_Brilla_tecnologia.csv', on_bad_lines='skip')
    df_control = pd.read_csv(path + '\data\BD_Brilla_control.csv', on_bad_lines='skip')

    archivos = [df_electrodomesticos, df_control, df_construccion, df_motos, df_tecnologia]

    numeric = ['CUPO_DISPONIBLE', 'ESTRATO', 'EDAD']
    categorical = ['GENERO', 'SEGMENTO', 'BARRIO', 'LOCALIDAD', 'DEPARTAMENTO']

    from Brilla_clase import main_brilla
    print('Con todas las bases de datos')
    pred1, predicciones_df_gases, Matriz_confusion = main_brilla(files=archivos,
                        numericas=numeric,
                        categoricas=categorical,
                        clasificador='random_forest',
                        matriz_confusion=True, 
                        df_gases=df_gases_caribe)
    
