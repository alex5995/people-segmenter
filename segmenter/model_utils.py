import sys
sys.path.insert(0, 'segmenter')

import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from model import ResNetUNet

img_trans = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])

mask_trans = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor()
])

def load_model(model_path, num_classes=1):
    
    model = ResNetUNet(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_mask(model, img):

    tensor_img = img_trans(img)
    with torch.no_grad():
        outputs = model(tensor_img.unsqueeze(0))
    
    reverse_mask = outputs.squeeze().detach().numpy() < 0

    img = np.array(T.ToPILImage()(mask_trans(img)).convert("RGBA"))
    img[reverse_mask,:] = [255,255,255,0]
    return Image.fromarray(img)
