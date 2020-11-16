import sys
sys.path.insert(0, 'segmenter')

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from dataset import get_transform
import torch
import numpy as np
from PIL import Image

predict_transform = get_transform(train=False)

def get_model_instance_segmentation(num_classes, pretrained=True):

    model = maskrcnn_resnet50_fpn(pretrained=pretrained, pretrained_backbone=pretrained)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def load_model(model_path, num_classes=2):
    
    model = get_model_instance_segmentation(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_mask(model, img):

    tensor_img, _ = predict_transform(img, img)
    outputs = model(tensor_img.unsqueeze(0))
    reverse_mask = outputs[0]['masks'][0].squeeze().detach().numpy() < 0.5

    img = np.array(img.convert('RGBA'))
    img[reverse_mask,:] = [255,255,255,0]
    return Image.fromarray(img)
