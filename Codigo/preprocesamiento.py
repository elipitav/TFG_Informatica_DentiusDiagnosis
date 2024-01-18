import time, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def nombre_directorio(patologias):
    directorio = ''
    for patologia in patologias:
        directorio +=  patologia
        directorio += '_'
    directorio = directorio[:-1]
    
    return directorio


def preparar_directorio_salida(patologias, nets):
    
    directorio = './../Salidas/' + nombre_directorio(patologias) + '/'

    # Creamos el directorio para la salida de la combinación de patologías si no existe ya
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    # Para cada combinación de patologías creamos un directorio de salidas para cada red
    for net in nets:
        red_dir = directorio + '/' + net + '/'
        if not os.path.exists(red_dir):
            # Creamos el directorio
            os.makedirs(red_dir)
    
    return directorio



def estratificacion(patologias, dataset_global = '/mnt/beegfs/groups/med_inf/dental_diagnosis/Dataset/anon.xlsx'):

    # Generamos el nombre del directorio
    directorio = './../Estratificacion/' + nombre_directorio(patologias) + '/'

    # Creamos el directorio y generamos los datasets si no existe la carpeta
    if not os.path.exists(directorio):
        
        # Creamos el directorio
        os.makedirs(directorio)

        # Leemos el excel
        labels = pd.read_excel(dataset_global)

        # Nos quedamos con el id de la imagen y con las patologias que nos interesan
        labels = labels.loc[:, ['opg_id'] + patologias]
        
        # Sustituimos los NaN por 0 y las X por 1
        labels.replace(to_replace= np.nan, value = 0, inplace=True)
        labels.replace(to_replace= 'X', value = 1, inplace=True)
        labels.replace(to_replace= 'x', value = 1, inplace=True)

        # Eliminamos las clases que tengan menos de 3 repeticiones ya que no serán representativas
        reps = labels.groupby(list(labels.columns[1:len(patologias)+1])).size().reset_index(name='count')
        labels = labels.merge(reps[reps['count'] >= 3], how = 'right')
        labels = labels.drop('count', axis = 1)

        # Separamos la columna con el índice de la imagen de las patologías
        X = labels.loc[:, 'opg_id']
        Y = labels.loc[:, patologias]

        # Realizamos la separacion estratificada
        # - 65% entrenamiento
        # - 20% validación
        # - 15% test
        semilla = int(time.time())
        X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, train_size=0.65, stratify=Y, random_state=semilla)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, train_size=0.43, stratify=Y_val_test, random_state=semilla)

        # Ahora volvemos a generar el dataframe completo para poder generar el excel de salida
        df_X_train = pd.DataFrame({'indice': X_train.index, 'opg_id': X_train.values})
        df_Y_train = pd.DataFrame({'indice': Y_train.index})
        for patologia in patologias:
            df_Y_train[patologia] = getattr(Y_train, patologia).values
        df_X_val = pd.DataFrame({'indice': X_val.index, 'opg_id': X_val.values})
        df_Y_val = pd.DataFrame({'indice': Y_val.index})
        for patologia in patologias:
            df_Y_val[patologia] = getattr(Y_val, patologia).values
        df_X_test = pd.DataFrame({'indice': X_test.index, 'opg_id': X_test.values})
        df_Y_test = pd.DataFrame({'indice': Y_test.index})
        for patologia in patologias:
            df_Y_test[patologia] = getattr(Y_test, patologia).values

        train = df_X_train.merge(df_Y_train) \
            .drop('indice', axis=1) \
            .sort_values('opg_id')
        val = df_X_val.merge(df_Y_val) \
            .drop('indice', axis=1) \
            .sort_values('opg_id')
        test = df_X_test.merge(df_Y_test) \
            .drop('indice', axis=1) \
            .sort_values('opg_id')

        # Generamos los ficheros de excel relativos a los datos
        train.to_excel(directorio+'/train.xlsx', index=False)
        val.to_excel(directorio+'/val.xlsx', index=False)
        test.to_excel(directorio+'/test.xlsx', index=False)

    return directorio
