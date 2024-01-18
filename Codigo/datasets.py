import os
import pandas as pd
import numpy as np
import torchvision.io as tvio
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

# Clase DatasetRadiografias
class DatasetRadiografias(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        # Leemos el excel
        self.img_labels = pd.read_excel(annotations_file)
        # Número de patologías estudiadas
        self.n_patologias = len(self.img_labels.columns)-1
        # Guardamos la ruta a la carpeta con las imágenes
        self.img_dir = img_dir
        # Transformaciones a aplicar a las imágenes
        self.transform = transform
        # Transformaciones a aplicar al resultado
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        pass

# Definimos la transformacion que aplicaremos a las imagenes en escala RGB
transform_three_channel = transforms.Compose(
    [transforms.ToPILImage(),
    transforms.Resize(size=(224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-5,5)),
    transforms.RandomAffine(translate=(0.1,0.1), scale=(0.95,1.05), degrees=(0,0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)


# Definimos la transformacion que aplicaremos a las imagenes en escala de grises
transform_one_channel = transforms.Compose(
    [transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.449,], std=[0.226,]) # Esto lo cambiamos por la media
    ]
)

# Dataset para trabajar con escala RGB
class DatasetThreeChannel(DatasetRadiografias):
    def __init__(self, annotations_file, img_dir, transform=transform_three_channel, target_transform=None):
        super().__init__(annotations_file, img_dir, transform, target_transform)
    
    def __getitem__(self, idx):
        # Obtenemos el indice de la imagen y le añadimos la extension '.jpg.'
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]) + '.jpg')
        # Leemos la imagen convirtiéndola a RGB
        image = tvio.read_image(img_path, tvio.ImageReadMode.RGB)
        # Obtenemos la etiqueta
        label = torch.DoubleTensor(self.img_labels.iloc[idx, 1:(self.n_patologias+1)])
        # Aplicamos las transformaciones
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# Dataset para trabajar con escala de grises
class DatasetOneChannel(DatasetRadiografias):
    def __init__(self, annotations_file, img_dir, transform=transform_one_channel, target_transform=None):
        super().__init__(annotations_file, img_dir, transform, target_transform)

    def __getitem__(self, idx):
        # Obtenemos el indice de la imagen y le añadimos la extension '.jpg.'
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]) + '.jpg')
        # Leemos la imagen convirtiéndola a RGB
        image = tvio.read_image(img_path, tvio.ImageReadMode.GRAY)
        # Obtenemos la etiqueta
        label = torch.DoubleTensor(self.img_labels.iloc[idx, 1:(self.n_patologias+1)])
        # Aplicamos las transformaciones
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
