from flask import Flask, request, render_template, jsonify
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import base64
from io import BytesIO
from model import CNNModel
import json  # Import JSON module

app = Flask(__name__)

# Pre-configured device and model settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNNModel().to(device)
checkpoint = torch.load('model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

labels = [
    'adonis', 'american snoot', 'an 88', 'banded peacock', 'beckers white',
    'black hairstreak', 'cabbage white', 'chestnut', 'clodius parnassian',
    'clouded sulphur', 'copper tail', 'crecent', 'crimson patch',
    'eastern coma', 'gold banded', 'great eggfly', 'grey hairstreak',
    'indra swallow', 'julia', 'large marble', 'malachite', 'mangrove skipper',
    'metalmark', 'monarch', 'orange oakleaf', 'orange tip',
    'orchard swallow', 'painted lady', 'paper kite', 'peacock', 'pine white',
    'pipevine swallow', 'purple hairstreak', 'question mark', 'red admiral',
    'red spotted purple', 'scarce swallow', 'silver spot skipper', 'sootywing',
    'southern dogface', 'straited queen', 'two barred flasher', 'ulyses',
    'viceroy', 'wood satyr', 'yellow swallow tail', 'zebra long wing'
]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed_image = transform(original_image)
            transformed_image = torch.unsqueeze(transformed_image, 0)

            with torch.no_grad():
                outputs = model(transformed_image.to(device))
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_probs, top_cats = probabilities.topk(5)

            top_labels = [labels[idx] for idx in top_cats[0]]
            top_confidences = top_probs[0].tolist()

            top_predictions = [(label, prob * 100) for label, prob in zip(top_labels, top_confidences)]

            buffered = BytesIO()
            img = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode('.jpg', img)
            image_base64 = base64.b64encode(buf).decode('utf-8')

            # Serialize data before sending to the template
            predictions_json = json.dumps({
                'top_labels': top_labels,
                'top_confidences': top_confidences
            })

            return render_template('index.html', prediction=top_labels[0],
                                   confidence=top_confidences[0] * 100,
                                   predictions_json=predictions_json,
                                   original_image=image_base64)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)