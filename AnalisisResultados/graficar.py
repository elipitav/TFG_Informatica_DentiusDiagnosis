import os
import argparse
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import torch
import pickle

path = '../Salidas/'

comparar_patologias = True
multiclase = True


def listar_objs(path, patologias, nets):
    """Función para obtener todos los resultados a representar en el caso de modelos uniclase

    Parámetros:
    - path (string): Ruta al directorio donde se encuentran todos los resultados
    - patologias (lista de strings): Lista con las patologías a analizar
    - nets (lista de strings): Lista con las redes a analizar

    Devuelve:
    - resultados (diccionario): Diccionario de listas. Cada entrada se asocia con una red o una patología,
    dependiendo de si se comparan patologías o redes, respectivamente. Cada una es una lista con todos
    los resultados asociados a esa red o patología. Cada resultado es un diccionario con los siguientes
    campos:
        - ruta: ruta al fichero pickle con los resultados serializados.
        - red: red asociada.
        - patologia: patología asociada.
        - ruta_excel: ruta al excel con los costes de entrenamiento y validación.
    
    """

    # Diccionario final con los resultados
    resultados = {}
    
    # Si comparamos patologías, cada entrada se asociará a una red. Y dentro de cada una se incluirán
    # los resultados para todas las patologías
    if comparar_patologias:
        for net in nets:
            resultados[net] = []
            for patologia in patologias:
                ruta_completa_patologia = os.path.join(path, patologia)
                if os.path.isdir(ruta_completa_patologia):
                    ruta_completa_net = os.path.join(ruta_completa_patologia, net)
                    if os.path.isdir(ruta_completa_net):
                        ruta_resultado = os.path.join(ruta_completa_net, 'metrics_output.pkl')
                        ruta_excel = os.path.join(ruta_completa_net, 'epoch_loss.xlsx')
                        if os.path.isfile(ruta_resultado) and os.path.isfile(ruta_excel):
                            resultado = {}
                            resultado['ruta'] = ruta_resultado
                            resultado['red'] = net
                            resultado['patologia'] = patologia
                            resultado['ruta_excel'] = ruta_excel
                            resultados[net].append(resultado)
    # Si comparamos redes, cada entrada se asociará a una patología. Y dentro de cada una se incluirán
    # los resultados para todas las redes
    else:
        for patologia in patologias:
            resultados[patologia] = []
            ruta_completa_patologia = os.path.join(path, patologia)
            if os.path.isdir(ruta_completa_patologia):
                for net in nets:
                        ruta_completa_net = os.path.join(ruta_completa_patologia, net)
                        if os.path.isdir(ruta_completa_net):
                            ruta_resultado = os.path.join(ruta_completa_net, 'metrics_output.pkl')
                            ruta_excel = os.path.join(ruta_completa_net, 'epoch_loss.xlsx')
                            if os.path.isfile(ruta_resultado) and os.path.isfile(ruta_excel):
                                resultado = {}
                                resultado['ruta'] = ruta_resultado
                                resultado['red'] = net
                                resultado['patologia'] = patologia
                                resultado['ruta_excel'] = ruta_excel
                                resultados[patologia].append(resultado)

    return resultados

def listar_objs_mult(path, patologias, nets):
    """Función para obtener todos los resultados a representar en el caso de modelos multiclase

    Parámetros:
    - path (string): Ruta al directorio donde se encuentran todos los resultados.
    - patologias (lista de strings): Lista con las patologías a analizar.
    - nets (lista de strings): Lista con las redes a analizar.

    Devuelve:
    - resultados (diccionario): Diccionario de listas. Cada entrada se asocia con una red o una patología,
    dependiendo de si se comparan patologías o redes, respectivamente. Cada una es una lista con todos
    los resultados asociados a esa red o patología. Cada resultado es un diccionario con los siguientes
    campos:
        - ruta: ruta al fichero pickle con los resultados serializados.
        - red: red asociada.
        - patologia: patología asociada.

    - resultados_entrenamiento (lista de diccionarios): Cada diccionario se compone de:
        - red: red asociada.
        - ruta_excel: ruta al excel con los costes de entrenamiento y validación.

    """
    
    # Diccionario final con los resultados
    resultados = {}
    
    # Creamos el nombre de la carpeta con las salidas, conteniendo el nombre de todas las patologías
    dir_name = patologias[0]
    for i in range(1, len(patologias)):
        dir_name = dir_name + '_' + patologias[i]
    ruta_completa_patologia = os.path.join(path, dir_name)

    if os.path.isdir(ruta_completa_patologia):
        # Si comparamos patologías, cada entrada se asociará a una red. Y dentro de cada una se incluirán
        # los resultados para todas las patologías
        if comparar_patologias:
            for net in nets:
                resultados[net] = []
                ruta_completa_net = os.path.join(ruta_completa_patologia, net)
                if os.path.isdir(ruta_completa_net):
                    for patologia in patologias:
                        ruta_resultado = os.path.join(ruta_completa_net, patologia + '_metrics_output.pkl')
                        if os.path.isfile(ruta_resultado):
                            resultado = {}
                            resultado['ruta'] = ruta_resultado
                            resultado['red'] = net
                            resultado['patologia'] = patologia
                            resultados[net].append(resultado)
        # Si comparamos redes, cada entrada se asociará a una patología. Y dentro de cada una se incluirán
        # los resultados para todas las redes
        else:
            for patologia in patologias:
                resultados[patologia] = []
                for net in nets:
                    ruta_completa_net = os.path.join(ruta_completa_patologia, net)
                    if os.path.isdir(ruta_completa_net):
                        ruta_resultado = os.path.join(ruta_completa_net, patologia + '_metrics_output.pkl')
                        if os.path.isfile(ruta_resultado):
                            resultado = {}
                            resultado['ruta'] = ruta_resultado
                            resultado['red'] = net
                            resultado['patologia'] = patologia
                            resultados[patologia].append(resultado)
    
    # Lista con los resultados del entrenamiento
    resultados_entrenamiento = []
    for net in nets:
        ruta_completa_net = os.path.join(ruta_completa_patologia, net)
        if os.path.isdir(ruta_completa_net):
            ruta_excel = os.path.join(ruta_completa_net, 'epoch_loss.xlsx')
            if os.path.isfile(ruta_excel):
                resultado_entrenamiento = {}
                resultado_entrenamiento['ruta_excel'] = ruta_excel
                resultado_entrenamiento['red'] = net
                resultados_entrenamiento.append(resultado_entrenamiento)

    
    return resultados, resultados_entrenamiento

def diagrama_barras(tipo_metrica, metrica, resultados):
    """Función para representar los resultados asociados a una única métrica mediante diagramas de barras. 
    En ella se compararán los resultados obtenidos para esa métrica en distintas combinaciones de redes y
    patologías.

    Parámetros:
    - tipo_metrica (string): Tipo de métrica a representar ('metricas_continuas' o 'metricas_binarias').
    - metrica (string): Métrica a representar.
    - resultados (lista de diccionarios): Lista con los diccionarios de los resultados que se quieren
    comparar.

    """

    # Dependiendo de que estemos comparando, tomaremos un orden u otro de variables
    if comparar_patologias:
        var1 = "red"
        var2 = "patologia"
    else:
        var1 = "patologia"
        var2 = "red"

    # Generamos el nombre del directorio de salida
    # Para ello escogemos entre el directorio multiclase o individual dependiendo del caso
    if multiclase:
        dir = 'Multiclase/' + resultados[0][var1] + '/'
    else:
        dir = 'Individual/' + resultados[0][var1] + '/'

    # Verificamos si el directorio existe y lo creamos si no existe
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Aquí guardaremos los datos de la gráfica
    data = {
        'Categoria': [],
        'Valor': []
    }

    # Obtenemos los resultados
    for resultado in resultados:
        with open(resultado['ruta'], 'rb') as fic:
            metricas = pickle.load(fic)
        data['Valor'].append(round(metricas[tipo_metrica][metrica], 2))
        data['Categoria'].append(resultado[var2] + '_' + resultado[var1])
    df = pd.DataFrame(data)

    # Creamos el gráfico de barras
    if multiclase:
        fig = px.bar(df, x='Categoria', y='Valor', text='Valor', title = metrica + ' (multiclase)')
    else:
        fig = px.bar(df, x='Categoria', y='Valor', text='Valor', title = metrica)

    # Guardamos el gráfico como archivo de imagen
    if multiclase:
        fig.write_image(dir + 'bar_' + metrica + '_multiclase.png')
    else:
        fig.write_image(dir + 'bar_' + metrica + '.png')

def diagrama_barras_varias_metricas(tipo_metrica, metricas, resultado):
    """Función para representar los resultados asociados a una varias métrica mediante diagramas de barras. 
    En ella se mostrarán los resultados obtenidos para varias métricas para la misma combinación de red y
    patología.

    Parámetros:
    - tipo_metrica (string): Tipo de métrica a representar ('metricas_continuas' o 'metricas_binarias').
    - metrica (string): Métrica a representar.
    - resultado (diccionario): Diccionario asociado a los resultados obtenidos para la combinación concreta
    de red y patología.

    """

    # Generamos el nombre del directorio de salida
    # Para ello escogemos entre el directorio multiclase o individual dependiendo del caso
    if multiclase:
        dir = 'Multiclase/' + resultado['red'] + '/'
    else:
        dir = 'Individual/' + resultado['red'] + '/'

    # Verificamos si el directorio existe y lo creamos si no existe
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Generamos el nombre del directorio para la patología a representar
    dir2 = dir + resultado['patologia'] + '/'

    # Verificamos si el directorio existe y lo creamos si no existe
    if not os.path.exists(dir2):
        os.makedirs(dir2)

    # Abrimos el archivo pickle
    with open(resultado['ruta'], 'rb') as fic:
        valores = pickle.load(fic)

    # Aquí guardaremos los datos de la gráfica
    data = {
        'Categoria': [],
        'Valor': []
    }

    # Obtenemos el resultado de cada métrica
    for metrica in metricas:
        data['Valor'].append(round(valores[tipo_metrica][metrica], 2))
        data['Categoria'].append(metrica)
    df = pd.DataFrame(data)

    # Creamos un gráfico de barras
    if multiclase:
        fig = px.bar(df, x='Categoria', y='Valor', text='Valor', title = resultado['patologia'] + ' con ' + resultado['red'] + ' (multiclase)')
    else:
        fig = px.bar(df, x='Categoria', y='Valor', text='Valor', title = resultado['patologia'] + ' con ' + resultado['red'])
        
    # Guardamos el gráfico como archivo de imagen
    if multiclase:
        fig.write_image(dir2 + 'multiclase_' + resultado['red'] + '_' + tipo_metrica + '_comp.png')
    else:
        fig.write_image(dir2 + resultado['red'] + '_' + tipo_metrica + '_comp.png')

def reprentar_matriz_confusion(resultado):

    """Función para representar la matriz de confusión asociada a una combinación de modelo y patología.

    Parámetros:
    - resultado (diccionario): Diccionario asociado a los resultados obtenidos para la combinación concreta
    de red y patología.

    """

    red = resultado['red']
    patologia = resultado['patologia']
    ruta = resultado['ruta']

    if multiclase:
        dir = os.path.join('Multiclase', red) 
    else:
        dir = os.path.join('Individual', red) 

    # Verificamos si el directorio existe y lo creamos si no existe
    if not os.path.exists(dir):
        os.makedirs(dir)

    dir = os.path.join(dir, patologia)

    # Verificamos si el directorio existe y lo creamos si no existe
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(ruta, 'rb') as fic:
        metricas = pickle.load(fic)
    
    # Obtenemos la matriz de confusión
    confusion_matrix = metricas['metricas_binarias']['confussion_matrix']

    # Invertimos el orden de las filas de la matriz de confusión
    confusion_matrix = confusion_matrix[::-1]     

    # Creamos la representación de la matriz
    class_labels = ['Negativo', 'Positivo']
    fig = ff.create_annotated_heatmap(
        z=confusion_matrix,
        x=class_labels,
        y=class_labels[::-1],
        colorscale='Blues',
        showscale=True
    )

    # Personalizamos el diseño
    fig.update_layout(
        title='Matriz de confusión para ' + patologia + ' con ' + red,
        xaxis=dict(title='Etiqueta Predicha'),
        yaxis=dict(title='Etiqueta Real'),
        margin=dict(l=150, r=50, b=50, t=150)
    )

    # Guardamos la figura
    ruta_imagen = os.path.join(dir, resultado['patologia'] + '_' + resultado['red'] +  '_confussion_matrix.png')
    fig.write_image(ruta_imagen)

def representar_coste_entrenamiento(resultado):

    """Función para representar una gráfica con la evolución del coste de validación y entrenamiento a lo
    largo de las distintas épocas.

    Parámetros:
    - resultado (diccionario): Diccionario asociado a los resultados obtenidos para la combinación concreta
    de modelo y patología.

    """

    red = resultado['red']
    
    # Creamos el directorio de salida dependiendo del caso
    if multiclase:
        dir = os.path.join('Multiclase', red)
    else:
        dir = os.path.join('Individual', red)
    
    # Verificamos si el directorio existe y lo creamos si no existe
        os.makedirs(dir)

    # Si no es multiclase habrá un resultado para cada patologia, por lo que creamos una carpeta para ella
    if not multiclase:
        dir = os.path.join(dir, resultado['patologia'])
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    # Leemos el archivo Excel
    df = pd.read_excel(resultado['ruta_excel'])

    # Extraemos los datos
    epochs = df.iloc[:, 0]  # Epocas
    train_cost = df.iloc[:, 1]  # Coste entrenamiento
    val_cost = df.iloc[:, 2]  # Coste validación

    # Dibujamos la variacíon para entrenamiento y validación
    plt.plot(epochs, train_cost, label='Coste del entrenamiento')
    plt.plot(epochs, val_cost, label='Coste de la validación')

    # Personalizamos las etiquetas de los ejes
    plt.xlabel(df.columns[0])  # Etiqueta del eje x
    plt.ylabel("Coste")  # Etiqueta del eje y

    # Añadimos título y leyenda
    if multiclase:
        plt.title(f'{red} para multiclase')
    else:
        patologia = resultado['patologia']
        plt.title(f'{red} para {patologia}')
    plt.legend()

    # Guardamos el gráfico
    ruta_imagen = os.path.join(dir, 'coste_entrenamiento.png')
    plt.savefig(ruta_imagen)




if __name__ == "__main__":
   
    # Creamos el parser para solicitar los argumentos
    parser = argparse.ArgumentParser(description='Graficadora dentius diagnosis')

    # Añadimos un argumento obligatorio que indicara el fichero json con las configuraciones a graficar
    parser.add_argument('-f', '--fichero', required=True, type=str, help='Fichero json de configuración')

    # Comprobamos que se introdujeron correctamente los argumentos y si no es así, mostramos la ayuda
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        print('\n')
        sys.exit(1)

    # Obtenemos la ruta al fichero json de configuración
    conf_path = args.fichero

    # Leemos el fichero de configuración 
    with open(conf_path, 'r') as archivo_json:
        configuraciones = json.load(archivo_json)

    # Para cada configuración obtenemos cada atributo
    for conf in configuraciones:
        patologias = conf['patologias']
        nets = conf['nets']
        # Indica si queremos ver una red frente a varias patologias (True) o al revés (False)
        if conf['compara'] == "patologias":
            comparar_patologias = True
        else:
            comparar_patologias = False
        # Indica si el modelo es multiclase o no
        if conf['multiclase'] == "Si":
            multiclase = True
        else:
            multiclase = False
        
        if multiclase:
            # Obtenemos las rutas a los archivos con los resultados
            resultados, resultados_entrenamiento = listar_objs_mult(path, patologias, nets)

            if comparar_patologias:
                for net in nets:
                    for metrica in conf['metricas_binarias']:
                        diagrama_barras('metricas_binarias', metrica, resultados[net])
                    for metrica in conf['metricas_continuas']:
                        diagrama_barras('metricas_continuas', metrica, resultados[net])
                    for resultado in resultados[net]:
                        diagrama_barras_varias_metricas('metricas_binarias', conf['metricas_binarias'], resultado)
                        diagrama_barras_varias_metricas('metricas_continuas', conf['metricas_continuas'], resultado)
                        reprentar_matriz_confusion(resultado)
            else:
                for patologia in patologias:
                    for metrica in conf['metricas_binarias']:
                        diagrama_barras('metricas_binarias', metrica, resultados[patologia])
                    for metrica in conf['metricas_continuas']:
                        diagrama_barras('metricas_continuas', metrica, resultados[patologia])
                    for resultado in resultados[patologia]:
                        diagrama_barras_varias_metricas('metricas_binarias', conf['metricas_binarias'], resultado)
                        diagrama_barras_varias_metricas('metricas_continuas', conf['metricas_continuas'], resultado)
                        reprentar_matriz_confusion(resultado)
            
            for resultado_entrenamiento in resultados_entrenamiento:
                representar_coste_entrenamiento(resultado_entrenamiento)
        
        else:
            resultados = listar_objs(path, patologias, nets)
            
            if comparar_patologias:
                for net in nets:
                    for metrica in conf['metricas_binarias']:
                        diagrama_barras('metricas_binarias', metrica, resultados[net])
                    for metrica in conf['metricas_continuas']:
                        diagrama_barras('metricas_continuas', metrica, resultados[net])
                    for resultado in resultados[net]:
                        diagrama_barras_varias_metricas('metricas_binarias', conf['metricas_binarias'], resultado)
                        diagrama_barras_varias_metricas('metricas_continuas', conf['metricas_continuas'], resultado)
                        reprentar_matriz_confusion(resultado)
                        representar_coste_entrenamiento(resultado)
            else:
                for patologia in patologias:
                    for metrica in conf['metricas_binarias']:
                        diagrama_barras('metricas_binarias', metrica, resultados[patologia])
                    for metrica in conf['metricas_continuas']:
                        diagrama_barras('metricas_continuas', metrica, resultados[patologia])
                    for resultado in resultados[patologia]:
                        diagrama_barras_varias_metricas('metricas_binarias', conf['metricas_binarias'], resultado)
                        diagrama_barras_varias_metricas('metricas_continuas', conf['metricas_continuas'], resultado)
                        reprentar_matriz_confusion(resultado)
                        representar_coste_entrenamiento(resultado)
                
        

