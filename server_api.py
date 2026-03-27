import torch
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import wikipedia
import io
import warnings

# 1. ELIMINARE WARNING BEAUTIFULSOUP
# Aceasta linie dezactivează avertismentul de parser fără a modifica librăria wikipedia
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

wikipedia.set_lang("ro")
app = Flask(__name__)
CORS(app)

# CONFIGURARE HARDWARE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DEFINIRE CLASE
class_names = [
    'catina', 'galbenele', 'ienupar', 'lavanda', 'lumanarica', 'maces', 'menta',
    'musetel', 'papadie', 'papalau', 'pelin', 'podbal', 'salcie', 'scaivanat',
    'soc', 'spanz', 'sunatoare', 'treifratipatati', 'volbura'
]

# 2. ÎNCĂRCARE MODEL (ARHITECTURĂ ROBUSTĂ)
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

# PREPROCESARE IMAGINE
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. SISTEM DE CACHING (MEMORIE LOCALĂ)
# Previne cererile inutile către API-ul Wikipedia pentru aceleași plante
wiki_cache = {}


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Nicio imagine trimisa'}), 400

    try:
        file = request.files['image']
        image_bytes = file.read()

        # Conversie RGB pentru a suporta PNG/RGBA fără erori
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # INFERENȚĂ
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top_prob, top_catid = torch.max(probabilities, 0)
        prediction = class_names[top_catid]
        siguranta_procent = top_prob.item() * 100

        # 4. PRAG DE ÎNCREDERE (CONFIDENCE THRESHOLD)
        # Dacă modelul este nesigur, returnăm un răspuns controlat
        if siguranta_procent < 58.0:
            return jsonify({
                'planta': 'IDENTIFICARE INCERTĂ',
                'siguranta': round(siguranta_procent, 2),
                'wikipedia': 'Modelul nu a putut identifica planta cu o precizie suficientă. Te rugăm să reîncerci cu o imagine mai clară sau dintr-un alt unghi.'
            })

        # 5. GESTIONARE DEFENSIVĂ WIKIPEDIA CU CACHING
        termen_cautare = prediction.replace("_", " ")

        if termen_cautare not in wiki_cache:
            try:
                # Limităm la 3 propoziții pentru un răspuns rapid în UI
                rezumat = wikipedia.summary(termen_cautare, sentences=3)
                wiki_cache[termen_cautare] = rezumat
            except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
                rezumat = f"Informațiile detaliate pentru {termen_cautare} nu au fost găsite, dar planta a fost identificată vizual."
                wiki_cache[termen_cautare] = rezumat
            except Exception:
                # Fallback în cazul în care pică internetul sau API-ul
                rezumat = "Informațiile botanice sunt momentan indisponibile (Eroare conexiune)."
        else:
            rezumat = wiki_cache[termen_cautare]

        return jsonify({
            'planta': prediction.upper(),
            'siguranta': round(siguranta_procent, 2),
            'wikipedia': rezumat
        })

    except Exception as e:
        # Returnăm eroarea în format JSON pentru a nu bloca Frontend-ul
        return jsonify({'error': f"Eroare server: {str(e)}"}), 500


@app.route('/get_info/<name>', methods=['GET'])
def get_info(name):
    """Rută pentru căutare manuală cu utilizarea cache-ului"""
    termen = name.lower().replace("_", " ")
    try:
        if termen in wiki_cache:
            rezultat = wiki_cache[termen]
        else:
            rezultat = wikipedia.summary(termen, sentences=3)
            wiki_cache[termen] = rezultat

        return jsonify({
            'planta': name.upper(),
            'wikipedia': rezultat
        })
    except:
        return jsonify({'error': 'Planta nu a fost găsită în baza de date Wikipedia.'}), 404


if __name__ == '__main__':
    print(f"[*] Server pornit pe dispozitivul: {device}")
    print(f"[*] Prag de încredere setat la: 58%")
    app.run(host='0.0.0.0', port=5000)