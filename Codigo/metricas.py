import ignite
import torch
from ignite.engine import Engine

class CustomEvaluator:
    def __init__(self, step, net, device, ind_patologia = None):
        self.net = net
        self.device = device
        self.ind_patologia = ind_patologia
        self.step = step

    def _evaluate_step(self, engine, batch):
        if self.ind_patologia is None:
            return self.step(engine, batch, self.net, self.device)
        else:
            return self.step(engine, batch, self.net, self.device, self.ind_patologia)

    def run(self, data_loader, test_metrics):
        evaluator = Engine(self._evaluate_step)

        # Registrar métricas para evaluar
        for name, metric in test_metrics.items():
            metric.attach(evaluator, name)

        state = evaluator.run(data_loader)

        return state

# Transformación necesaria para poder generar la matriz de confusión
def binary_one_hot_output_transform(output):
    y_pred, y = output
    y_pred = ignite.utils.to_onehot(y_pred.round().long(), 2)
    y = y.long()
    return y_pred, y

# Con esta parte trabajamos con métricas para las que no necesitamos un resultado binario
def test_step(engine, batch, net, device, ind_patologia = None):
    with torch.no_grad():
        if ind_patologia is None:
            x, y = batch[0].to(device), batch[1].to(device)
            y_pred = net.model(x)
            y_pred = torch.sigmoid(y_pred)
            return y_pred, y
        else:
            x, y = batch[0].to(device), batch[1].to(device)
            y_pred = net.model(x)
            y_pred = torch.sigmoid(y_pred)
            return y_pred[:,ind_patologia], y[:,ind_patologia]

# Con esta parte trabajamos con métricas para las que necesitamos un resultado binario, por tanto aproximamos
def binary_test_step(engine, batch, net, device, ind_patologia = None):
    with torch.no_grad():
        if ind_patologia is None:
            x, y = batch[0].to(device), batch[1].to(device)
            y_pred = net.model(x)
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.round()
            return y_pred, y
        else:
            x, y = batch[0].to(device), batch[1].to(device)
            y_pred = net.model(x)
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.round()
            return y_pred[:,ind_patologia], y[:,ind_patologia]