import os, sys
import torch
import numpy as np
import pandas as pd
from metricas import CustomEvaluator
import time
import pickle

# Metricas
import ignite
from ignite.engine import Engine
from ignite.metrics import Accuracy, Loss
from ignite.metrics.precision import Precision
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.metrics.recall import Recall
from ignite.contrib.metrics import ROC_AUC

import metricas

def entrenamiento(net, device, data_path, out_path, batch_size, max_epochs, max_epochs_without_improvement, img_path = '/mnt/beegfs/groups/med_inf/dental_diagnosis/Dataset/img_jpg'):

    # Comenzamos el entrenamiento + validación
    print('Comenzando entrenamiento...')

    # Guarda el antiguo valor de sys.stdout
    old_stdout = sys.stdout

    # Redirigimos la salida a un archivo de 
    archivo_salida = open(out_path + net.get_name() + '/train_output.txt', 'w')
    sys.stdout = archivo_salida

    # Para medir el tiempo de ejecucion
    total_time = 0

    # Inicializamos el coste mínimo en infinito
    min_valid_loss = np.inf
    epochs_without_improvement = 0

    epochs_loss = pd.DataFrame(columns=['Epoca', 'Coste'])

    net.model = net.model.to(device)

    train_output_path = out_path + net.get_name() + '/saved_model.pth'
    if os.path.exists(train_output_path):
        net.model.load_state_dict(torch.load(train_output_path))

    # Generamos los datasets
    train_set = net.dataset_type(annotations_file = data_path + 'train.xlsx', img_dir = img_path)
    val_set = net.dataset_type(annotations_file = data_path + 'val.xlsx', img_dir = img_path)

    # Cargamos los iteradores para poder trabajar con ellos
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Realizamos varias epocas de entramiento a no ser que a través de los datos de validacion tengamos un coste menor a 1
    for epoch in range(max_epochs): 

        print(f'#####Epoca {epoch+1}#####')

        train_loss = 0.0

        start_time = time.time()

        for i, data in enumerate(train_loader, 0):
            # Obtenemos los datos de entrada; una lista de [datos, etiquetas]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Inicializamos los parametros del gradiente
            net.optimizer.zero_grad()

            # Forward + Backward + Optimizar
            outputs = net.model(inputs)
            loss = net.criterion(outputs, labels)
            loss.backward()
            net.optimizer.step()

            # Mostramos las estadísticas asociadas al coste
            train_loss += loss.item()
            if i % 50 == 49:  # Mostramos cada 50
                print(f'[{epoch + 1}, {i + 1:5d}] Coste: {train_loss / 50:.3f}')
                train_loss = 0.0
  
        val_loss = 0.0
        for i, data in enumerate(val_loader, 0):
            # Obtenemos los datos de entrada; una lista de [datos, etiquetas]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Calculamos el coste
            outputs = net.model(inputs)
            loss = net.criterion(outputs,labels)
            val_loss += loss.item()

        # Calculamos el coste en la validación y lo añadimos al dataframe
        val_loss = val_loss/len(val_loader)
        epochs_loss.loc[epoch]=[epoch, val_loss]

        # Calculamos el tiempo de la época
        end_time = time.time()
        epoch_time = end_time - start_time
        total_time += epoch_time

        # Comprobamos si el coste en validación se redujo y si ocurre eso, guardamos el modelo
        print(f'Coste medio en la validación: {val_loss}\nTiempo: {epoch_time} s')
        if min_valid_loss > val_loss:
            print(f'El coste total en validacion se redujo({min_valid_loss:.6f}--->{val_loss:.6f}) \t Guardando el modelo')
            min_valid_loss = val_loss
            # Guardamos el estado
            torch.save(net.model.state_dict(), train_output_path)
            epochs_without_improvement = 0
        # Si no, comprobamos cuantas llevamos sin mejorar
        else:
            epochs_without_improvement+=1
            if epochs_without_improvement >= max_epochs_without_improvement:
                print(f'{max_epochs_without_improvement} epocas sin mejoras. Finalización en la época: {epoch}')
                break
        print("######################")
    
    # Mostramos el tiempo total de entrenamiento y validación
    print('##### RESUMEN #####')
    print(f'\tTiempo total de entrenamiento y validación: {total_time} s')
    
    # Y la media por época
    mean_time = total_time / (epoch+1)
    print(f'\tTiempo medio de entrenamiento y validación por época: {mean_time} s')
    
    # Al acabar generamos el excel con los datos
    epochs_loss.to_excel(out_path + net.get_name() + '/epoch_loss.xlsx', index=False)

    # Restaura sys.stdout al valor original fuera del bloque with
    sys.stdout = old_stdout
    
    # Cerramos el archivo de salida
    archivo_salida.close()

    print('Entrenamiento acabado.')


def test(net, device, data_path, out_path, batch_size, img_path = '/mnt/beegfs/groups/med_inf/dental_diagnosis/Dataset/img_jpg'):
    # Comenzamos la prueba
    print('Comenzando prueba...')

    # Guarda el antiguo valor de sys.stdout
    old_stdout = sys.stdout

    # Redirigimos la salida a un archivo de texto
    archivo_salida = open(out_path + net.get_name() + '/test_output.txt', 'w')
    sys.stdout = archivo_salida

    # Generamos el nombre del archivo pickle donde almacenaremos el resultado serializado
    objeto_salida = out_path + net.get_name() + '/metrics_output.pkl'

    net.model = net.model.to(device)
    
    train_output_path = out_path + net.get_name() + '/saved_model.pth'
    if os.path.exists(train_output_path):
        net.model.load_state_dict(torch.load(train_output_path))

    # Generamos el datasets
    test_set = net.dataset_type(annotations_file = data_path + 'test.xlsx', img_dir = img_path)

    # Cargamos el iterador para poder trabajar con el dataset
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # Si trabajamos con una clase, unicamente necesitaremos realizar dos grupos de métricas, binarias y continuas
    if net.num_classes == 1:
        # Variable para serializar los resultados
        resultados = {
            "metricas_continuas": {},
            "metricas_binarias": {}
        }

        # Usamos el paso que unicamente aplicar la función sigmoide
        test_evaluator = CustomEvaluator(metricas.test_step, net = net, device = device)

        # Métricas continuas
        test_metrics = {
            "loss": Loss(loss_fn=net.criterion, device=device),
            "confussion matrix": ConfusionMatrix(num_classes=2, device=device, 
                                                 output_transform=metricas.binary_one_hot_output_transform),
            "auc": ROC_AUC(device=device)
        }

        state = test_evaluator.run(test_loader, test_metrics)
        resultados['metricas_continuas'] = state.metrics
        print(f'Resultados métricas: {state.metrics}')

        # Usamos el paso para transformar a binario
        test_evaluator = CustomEvaluator(metricas.binary_test_step, net = net, device = device)

        # Métricas binarias
        binary_test_metrics = {
            "accuracy": Accuracy(device=device),
            "precision": Precision(device=device),
            "recall": Recall(device=device)
        }

        state = test_evaluator.run(test_loader, binary_test_metrics)
        resultados['metricas_binarias'] = state.metrics
        print(f'Resultados métricas binarias: {state.metrics}')

        with open(objeto_salida, 'wb') as obj:
            pickle.dump(resultados, obj)

                
    # Si hay más de una clase, calculamos el coste y accuracy globalmente
    # Pero también necesitaremos realizar métricas de 1 vs el resto
    else:
        # Variable para serializar los resultados
        resultados = {
            "metricas_globales": {
                "metricas_continuas": {},
                "metricas_binarias": {}
            }
        }

        # Usamos el paso que unicamente aplicar la función sigmoide
        test_evaluator = CustomEvaluator(metricas.test_step, net = net, device = device)

        # Métricas globales continuas
        test_metrics = {
            "loss": Loss(loss_fn=net.criterion, device=device)
        }

        state = test_evaluator.run(test_loader, test_metrics)
        resultados['metricas_globales']['metricas_continuas'] = state.metrics
        print(f'Resultado global métricas: {state.metrics}')
                
        test_evaluator = CustomEvaluator(metricas.binary_test_step, net = net, device = device)

        # Métricas globales discretas
        binary_test_metrics = {
            "accuracy": Accuracy(device=device, is_multilabel=True)
        }

        state = test_evaluator.run(test_loader, binary_test_metrics)
        resultados['metricas_globales']['metricas_binarias'] = state.metrics
        print(f'Resultados global métricas binarias: {state.metrics}')

        # Para cada clase generamos una transformación binaria y otra continua que unicamente coja una columna
        for i in range(net.num_classes):

            resultados[net.patologias[i]] = {
                "metricas_continuas": {},
                "metricas_binarias": {}
            }

            test_evaluator = CustomEvaluator(metricas.test_step, net = net, device = device, 
                                             ind_patologia = i)

            test_metrics = {
                "auc": ROC_AUC(device=device)
            }
                    
            state = test_evaluator.run(test_loader, test_metrics)
            resultados[net.patologias[i]]['metricas_continuas'] = state.metrics
            print(f'Resultados patologia {i} métricas: {state.metrics}')

            test_evaluator = CustomEvaluator(metricas.binary_test_step, net = net, device = device, 
                                             ind_patologia = i)
                    
            binary_test_metrics = {
                "accuracy": Accuracy(device=device),
                "precision": Precision(device=device),
                "recall": Recall(device=device)
            }

            state = test_evaluator.run(test_loader, binary_test_metrics)
            resultados[net.patologias[i]]['metricas_binarias'] = state.metrics
            print(f'Resultados patologia {i} métricas binarias: {state.metrics}')
        
        with open(objeto_salida, 'wb') as obj:
            pickle.dump(resultados, obj)
    
    # Restaura sys.stdout al valor original fuera del bloque with
    sys.stdout = old_stdout
    
    # Cerramos el archivo de salida
    archivo_salida.close()

    print('Prueba acabada')