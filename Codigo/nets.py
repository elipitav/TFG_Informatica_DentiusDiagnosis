import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import sys
import datasets

# Método para obtener la clase seleccionada
def seleccionar_red(nombre_red):
    # Obtén la clase según el nombre proporcionado
    clase_seleccionada = getattr(sys.modules[__name__], nombre_red)
    return clase_seleccionada

class Resnet18_OneChannel():
    def __init__(self, patologias):
        self.num_classes = len(patologias)
        self.patologias = patologias
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.model.fc = nn.Linear(512, self.num_classes)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.dataset_type = datasets.DatasetOneChannel
    
    def get_name(self):
        return 'Resnet18_OneChannel'
    
class Resnet18_ThreeChannel():
    def __init__(self, patologias):
        self.num_classes = len(patologias)
        self.patologias = patologias
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.model.fc = nn.Linear(512, self.num_classes)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.dataset_type = datasets.DatasetThreeChannel
    
    def get_name(self):
        return 'Resnet18_ThreeChannel'

class Resnet34_ThreeChannel():
    def __init__(self, patologias):
        self.num_classes = len(patologias)
        self.patologias = patologias
        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, self.num_classes)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.dataset_type = datasets.DatasetThreeChannel
    
    def get_name(self):
        return 'Resnet34_ThreeChannel'

class Resnet50_ThreeChannel():
    def __init__(self, patologias):
        self.num_classes = len(patologias)
        self.patologias = patologias
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, self.num_classes)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.dataset_type = datasets.DatasetThreeChannel
    
    def get_name(self):
        return 'Resnet50_ThreeChannel'