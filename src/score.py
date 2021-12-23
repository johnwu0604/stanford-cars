import torch
import json
import os
import requests
from model import ResNet152
from PIL import Image
from torchvision import transforms

def init():
    global model, model_dir, headers
    model_dir = os.environ['AZUREML_MODEL_DIR'] + '/outputs'
    headers = requests.utils.default_headers()
    headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
    model = ResNet152()
    model.load_state_dict(torch.load(model_dir + '/model.pth'))
    model.eval()

def run(raw_data):
    image_url = json.loads(raw_data)['image_url']
    with open('temp.jpg', 'wb') as f:
        download = requests.get(image_url, headers=headers)
        f.write(download.content)
    image = Image.open('temp.jpg').convert('RGB')

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=True)
        ])
    image_preprocessed = preprocess(image)
    batch_image_tensor = torch.unsqueeze(image_preprocessed, 0)

    output = model(batch_image_tensor)
    _, index = torch.max(output, 1)

    pred = ''
    with open(model_dir + '/classes.json') as f:
        classes = json.load(f)
        for item in classes:
            if classes[item] == int(index[0]):
                pred = item
    return {'prediction': pred}
