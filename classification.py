import streamlit as st
from PIL import Image
import os
import warnings
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn.functional as F

warnings.filterwarnings('ignore')

classes = ['glass', 'metal', 'organic', 'paper', 'plastic']
def build_model():
    model = efficientnet_b0(EfficientNet_B0_Weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(classes))
    return model

folder_path = os.path.dirname(__file__)
classifier = build_model()
classifier_path = os.path.join(folder_path, 'classifier.pth')
classifier.load_state_dict(torch.load(classifier_path, map_location=torch.device('cpu')))
classifier.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.set_page_config('EcoAI Waste Classifier', ':recycle:')
st.title('EcoAI Waste Classifier :recycle:')
img_file = st.camera_input('Classifying into one of these categories: cardboard, glass, metal, paper, plastic')

if img_file:
    img = Image.open(img_file)
    input = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        output = classifier(input)
        probs = F.softmax(output, dim=1)
        _, pred = torch.max(probs, dim=1)
        idx = pred.item()
        label = classes[idx]
        conf = round(probs[0][idx].item(), 2)
        st.info(f'{label}, Confidence: {conf*100} %')
