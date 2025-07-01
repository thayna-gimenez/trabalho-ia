import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import os
from PIL import Image

# Aumento de dados e normalização para treinamento
data_transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Normalização para validação
data_transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Carregando o dataset
# Classe para nosso próprio dataset com as classes Papel, plástico, metal e vidro
class LinhasDataset(Dataset):
    def __init__(self, img_dir, split, transform):
        self.classes = ['paper', 'plastic', 'metal', 'glass']

        self.split_path = os.path.join(img_dir, split)
        self.split_path_c0 = os.path.join(self.split_path, self.classes[0])
        self.split_path_c1 = os.path.join(self.split_path, self.classes[1])
        self.split_path_c2 = os.path.join(self.split_path, self.classes[2])
        self.split_path_c3 = os.path.join(self.split_path, self.classes[3])

        self.img_paths_c0 = [os.path.join(self.split_path_c0, f) for f in os.listdir(self.split_path_c0) if f.endswith('jpg')]
        self.img_labels_c0 = [0] * len(self.img_paths_c0)

        self.img_paths_c1 = [os.path.join(self.split_path_c1, f) for f in os.listdir(self.split_path_c1) if f.endswith('jpg')]
        self.img_labels_c1 = [1] * len(self.img_paths_c0)

        self.img_paths_c2 = [os.path.join(self.split_path_c2, f) for f in os.listdir(self.split_path_c2) if f.endswith('jpg')]
        self.img_labels_c2 = [2] * len(self.img_paths_c0)

        self.img_paths_c3 = [os.path.join(self.split_path_c3, f) for f in os.listdir(self.split_path_c3) if f.endswith('jpg')]
        self.img_labels_c3 = [3] * len(self.img_paths_c0)

        self.img_paths = self.img_paths_c0 + self.img_paths_c1 + self.img_paths_c2 + self.img_paths_c3
        self.img_labels = self.img_labels_c0 + self.img_labels_c1 + self.img_labels_c2 + self.img_labels_c3

        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        image = self.transform(image)

        label = self.img_labels[idx]

        return image, label

# Definindo o caminho do dataset
img_dir = './dataset'
image_dataset_train = LinhasDataset(img_dir=img_dir, split='train', transform=data_transforms_train)
image_dataset_val = LinhasDataset(img_dir=img_dir, split='val', transform=data_transforms_val)
image_dataset_test = LinhasDataset(img_dir=img_dir, split='test', transform=data_transforms_val)

# Criando o loader
dataloader_train = torch.utils.data.DataLoader(image_dataset_train, batch_size=4, shuffle=True, num_workers=2)
dataloader_val = torch.utils.data.DataLoader(image_dataset_val, batch_size=4, shuffle=True, num_workers=2)
dataloader_test = torch.utils.data.DataLoader(image_dataset_test, batch_size=4, shuffle=True, num_workers=2)

dataset_size_train = len(image_dataset_train)
dataset_size_val = len(image_dataset_val)
dataset_size_test = len(image_dataset_test)

class_names = image_dataset_train.classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Visualizando imagens
# Não funciona
#def imshow(inp, title=None):
    #"""Display image for Tensor."""
    #inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    #inp = np.clip(inp, 0, 1)
    #plt.imshow(inp)
    #if title is not None:
        #plt.title(title)
    #plt.pause(0.001)  # pausar um pouco


# Obtendo um batch do conjunto de treino
#inputs, classes = next(iter(dataloader_train))

#out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])


