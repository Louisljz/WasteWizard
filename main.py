import streamlit as st
from PIL import Image
import os
import warnings
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
import torch.nn.functional as F
from ultralytics import YOLO

warnings.filterwarnings('ignore')

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
def build_model():
    model = mobilenet_v2()
    model.classifier[-1] = nn.Linear(model.last_channel, len(classes))
    return model

folder_path = os.path.dirname(__file__)
detector_path = os.path.join(folder_path, 'yolov8n.pt')
detector = YOLO(detector_path)
classifier = build_model()
classifier_path = os.path.join(folder_path, 'classifier.pth')
classifier.load_state_dict(torch.load(classifier_path, map_location=torch.device('cpu')))
classifier.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.set_page_config('Trash Classifier', ':recycle:')
st.title('Trash Classifier :recycle:')
img_file = st.camera_input('Trash to Classify')

if img_file:
    img = Image.open(img_file)
    results = detector.predict(img, verbose=False)

    for r in results:    
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            cropped_img = img.crop(b.tolist())
            input = preprocess(cropped_img).unsqueeze(0)

            with torch.no_grad():
                output = classifier(input)
                probs = F.softmax(output, dim=1)
                _, pred = torch.max(probs, dim=1)
                idx = pred.item()
                label = classes[idx]
                conf = round(probs[0][idx].item(), 2)

                if conf > 0.6:
                    st.image(cropped_img, label)
                else:
                    st.info('AI cannot detect/classify trash. Mark as Others.')
