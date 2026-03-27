import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# Detecție GPU [cite: 3, 29]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = r'C:\Users\theda\PycharmProjects\AI_plante\data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Arhitectura 2: MobileNetV2 (Eficientă și rapidă)
model = models.mobilenet_v2(weights='DEFAULT')

# Înghețăm straturile inferioare [cite: 25]
for param in model.parameters():
    param.requires_grad = False

# Modificăm clasificatorul final
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# Folosim SGD cu momentum pentru diversitate față de Adam-ul tău original
optimizer = optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9)

# Codul de train simplificat [cite: 17, 18]
def train_alt():
    for epoch in range(5): # Doar 5 epoci, e "la mișto"
        print(f'Epoch {epoch+1}/5')
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
        print("Finalizată epoca.")
    torch.save(model.state_dict(), 'mobilenet_model.pth')

if __name__ == '__main__':
    train_alt()