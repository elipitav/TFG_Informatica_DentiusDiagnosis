import json
import sys
import argparse
import preprocesamiento as prep
import train_and_test as tat
import nets as nets
import torch

if __name__ == '__main__':

    # Creamos el parser para solicitar los argumentos
    parser = argparse.ArgumentParser(description='Lanzador dentius diagnosis')

    # Creamos un grupo mutuamente exclusivo, de forma que los argumentos en el sean excluyentes entre si
    # A traves de el indicaremos si queremos entrenar o probar
    group = parser.add_mutually_exclusive_group()
    # Indicamos que el grupo es obligatorio, hay necesidad de introducir alguno de ellos
    group.required = True
    # En ambos indicamos action como store_true ya que solo nos interesa saber si se introduce esa opcion o no
    group.add_argument('-t', '--train', action='store_true', help='Especifica que se quiere entrenar')
    group.add_argument('-p', '--test', action='store_true', help='Especifica que se quiere probar')
    # Anhadimos un argumento obligatorio que indicara el fichero json con las configuraciones a probar
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

    # Intentamos usar GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Si se quiere entrenar
    if args.train:
        # Para cada combinación de patologías especificada:
        # 1. Preparamos el directorio con los conjuntos de entrenamiento, salida y test
        # 2. Preparamos el directorio donde se añadirá la salida del modelo entrenado
        # 3. Comenzamos el entrenamiento
        for conf in configuraciones:
            patologias = conf['patologias']
            print(f'Entrenamiento con las patologias: {patologias}')
            data_path = prep.estratificacion(patologias)
            out_path = prep.preparar_directorio_salida(patologias, conf['nets'])
            for net_name in conf['nets']:
                print(f'Usando la red {net_name}')
                net = nets.seleccionar_red(net_name)(patologias)
                tat.entrenamiento(net, device, data_path, out_path, 
                conf['batch_size'], conf['max_epochs'], conf['max_epochs_without_improvement'])
    else:
        for conf in configuraciones:
            patologias = conf['patologias']
            print(f'Prueba con las patologias: {patologias}')
            data_path = prep.estratificacion(patologias)
            out_path = prep.preparar_directorio_salida(patologias, conf['nets'])
            for net_name in conf['nets']:
                print(f'Usando la red {net_name}')
                net = nets.seleccionar_red(net_name)(patologias)
                tat.test(net, device, data_path, out_path, conf['batch_size'])