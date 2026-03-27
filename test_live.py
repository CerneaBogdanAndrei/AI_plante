import torch
from torchvision import models, transforms
from PIL import Image
import os
import wikipedia

wikipedia.set_lang("ro")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = r'C:\Users\theda\PycharmProjects\AI_plante\data\train'
class_names = sorted(os.listdir(data_dir))

model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(512, len(class_names))
)

model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top_prob, top_catid = torch.max(probabilities, 0)
        prediction = class_names[top_catid]

        print(f"\n[{prediction.upper()}] - Siguranta: {top_prob.item() * 100:.2f}%\n")

        try:
            termen_cautare = prediction.replace("_", " ")
            rezumat = wikipedia.summary(termen_cautare, sentences=3)
            print(f"--- INFO WIKIPEDIA ---\n{rezumat}\n----------------------\n")
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"--- INFO WIKIPEDIA ---\nMai multe rezultate gasite. Fiti mai specific.\n----------------------\n")
        except wikipedia.exceptions.PageError:
            print(
                f"--- INFO WIKIPEDIA ---\nNu am gasit pagina pe Wikipedia pentru {prediction}.\n----------------------\n")

    except Exception as e:
        print(f"Eroare: {e}")


if __name__ == '__main__':
    while True:
        cale_poza = input("Trage aici poza cu planta (sau scrie 'stop' pt a iesi): ")
        if cale_poza.lower() == 'stop':
            break
        cale_poza = cale_poza.strip('\'"')
        predict_image(cale_poza)