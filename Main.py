import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import io

# Setup
cudnn.benchmark = True
plt.ion()  # interactive mode
st.set_option('deprecation.showPyplotGlobalUse', False)

# Data transformations
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

# Load dataset
data_dir = 'chest_xray'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Helper function to show images
def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Streamlit UI
st.title("Sistema de Classificação Diagnóstico")

# Show sample training images
st.header("Imagens Classificadas")
inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
plt.figure()
imshow(out, title=[class_names[x] for x in classes])
st.pyplot()

# Initialize model
model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, len(class_names))
model_conv = model_conv.to(device)

# Prediction function
def predict_image(model, image_bytes):
    try:
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Apply transformations
        transform = data_transforms['val']
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)
        
        return class_names[preds[0]], img
    
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

# File upload and prediction
col1, col2 = st.columns(2)

with col1:
    st.header("Faça o Upload da Imagem de Diagnóstico")
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

with col2:
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Imagem Carregada", use_column_width=True)
        
        # Make prediction
        prediction, img = predict_image(model_conv, uploaded_file.getvalue())
        
        if prediction is not None:
            st.success(f"Predição: {prediction}")
            
            # Show prediction with matplotlib
            transform = data_transforms['val']
            img_tensor = transform(img).unsqueeze(0)
            
            plt.figure()
            imshow(img_tensor.cpu().data[0], title=f'Predicted: {prediction}')
            st.pyplot()
