import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = r'C:\Users\theda\PycharmProjects\AI_plante\data\val'
model_path = 'best_model.pth'

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, data_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
class_names = dataset.classes

model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(512, len(class_names))
)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Greens')
plt.xlabel('Predictie AI')
plt.ylabel('Adevar')
plt.title('Matricea de Confuzie - Severin Bumbaru')
plt.savefig('matrice_confuzie.png')
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))