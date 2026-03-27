import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = r'C:\Users\theda\PycharmProjects\AI_plante\data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Arhitectura 3: VGG16 (Robustă, dar lentă)
model = models.vgg16(weights='DEFAULT')

for param in model.features.parameters():
    param.requires_grad = False

# Adaptăm clasificatorul final
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# Learning rate mai mic pentru VGG
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0005)

def train_vgg():
    for epoch in range(3):
        print(f'Epoch {epoch+1}/3')
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
        print("Epocă terminată.")
    torch.save(model.state_dict(), 'vgg_model.pth')

if __name__ == '__main__':
    train_vgg()