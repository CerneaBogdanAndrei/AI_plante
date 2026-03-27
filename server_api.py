import torch
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import wikipedia
import io

# Configurare Wikipedia în limba română
wikipedia.set_lang("ro")

app = Flask(__name__)

# CONFIGURARE CORS: Permite browserului să vorbească cu serverul fără blocaje
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,ngrok-skip-browser-warning')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


# Activare GPU (RTX 3070)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Speciile antrenate
class_names = [
    'catina', 'galbenele', 'ienupar', 'lavanda', 'lumanarica', 'maces', 'menta',
    'musetel', 'papadie', 'papalau', 'pelin', 'podbal', 'salcie', 'scaivanat',
    'soc', 'spanz', 'sunatoare', 'treifratipatati', 'volbura'
]

# Încărcare Model ResNet50
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

# Transformări pentru imagine
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return make_response()

    if 'image' not in request.files:
        return jsonify({'error': 'Nicio imagine trimisa'}), 400

    try:
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top_prob, top_catid = torch.max(probabilities, 0)
        prediction = class_names[top_catid]

        try:
            rezumat = wikipedia.summary(prediction.replace("_", " "), sentences=3)
        except:
            rezumat = f"Informații botanice indisponibile pentru {prediction}."

        return jsonify({
            'planta': prediction.upper(),
            'siguranta': round(top_prob.item() * 100, 2),
            'wikipedia': rezumat
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_info/<name>', methods=['GET'])
def get_info(name):
    try:
        rezumat = wikipedia.summary(name.replace("_", " "), sentences=3)
        return jsonify({'planta': name.upper(), 'siguranta': 100.0, 'wikipedia': rezumat})
    except:
        return jsonify({'error': 'Nu am găsit informații'}), 404


if __name__ == '__main__':
    print(f"✅ Server AI ACTIV pe: {device}")
    app.run(host='0.0.0.0', port=5000)