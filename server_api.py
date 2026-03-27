import torch
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import wikipedia
import io

# Configurare Wikipedia în limba română [cite: 1023]
wikipedia.set_lang("ro")

app = Flask(__name__)

# MODIFICARE: Configurare CORS mai robustă pentru a accepta cereri din orice sursă
CORS(app, resources={r"/*": {"origins": "*"}})

# Detectare dispozitiv (GPU dacă e disponibil, altfel CPU) [cite: 1023]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cele 19 specii de plante medicinale [cite: 1023]
class_names = [
    'catina', 'galbenele', 'ienupar', 'lavanda', 'lumanarica', 'maces', 'menta',
    'musetel', 'papadie', 'papalau', 'pelin', 'podbal', 'salcie', 'scaivanat',
    'soc', 'spanz', 'sunatoare', 'treifratipatati', 'volbura'
]

# Încărcarea arhitecturii modelului ResNet50 [cite: 1023]
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(512, len(class_names))
)

# Încărcarea greutăților antrenate (best_model.pth) [cite: 1023]
model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# Pipeline de preprocesare pentru imagini [cite: 1023, 1024]
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Nicio imagine trimisa'}), 400

    file = request.files['image']
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Pregătirea imaginii pentru rețeaua neuronală
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # Rularea predicției AI [cite: 1025]
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_prob, top_catid = torch.max(probabilities, 0)
    prediction = class_names[top_catid]
    siguranta = top_prob.item() * 100

    # Preluare informații de pe Wikipedia [cite: 1025, 1026]
    try:
        termen_cautare = prediction.replace("_", " ")
        rezumat = wikipedia.summary(termen_cautare, sentences=3)
    except:
        rezumat = f"Informațiile botanice pentru {prediction} sunt în curs de actualizare pe Wikipedia."

    return jsonify({
        'planta': prediction.upper(),
        'siguranta': round(siguranta, 2),
        'wikipedia': rezumat
    })

# Ruta pentru corectarea manuală a speciei [cite: 1026, 1027]
@app.route('/get_info/<name>', methods=['GET'])
def get_info(name):
    try:
        termen_cautare = name.replace("_", " ")
        rezumat = wikipedia.summary(termen_cautare, sentences=3)
        return jsonify({
            'planta': name.upper(),
            'siguranta': 100.0,
            'wikipedia': rezumat
        })
    except:
        return jsonify({'error': 'Nu s-au găsit informații pe Wikipedia pentru această plantă.'}), 404

if __name__ == '__main__':
    print(f"Server AI pornit pe: {device}")
    app.run(host='0.0.0.0', port=5000)