import os
os.system('python segmenter/download_model.py')

import sys
sys.path.insert(0, 'segmenter')

from fastapi import FastAPI, File, UploadFile, Response
from PIL import Image
from io import BytesIO
from model_utils import load_model, predict_mask

app = FastAPI()
model = load_model('./model.bin')

def read_image_file(file) -> Image.Image:
    img = Image.open(BytesIO(file)).convert("RGB")
    return img
    
def get_image_bytes(img):
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

@app.post("/segmentation")
async def predict_api(file: UploadFile = File(...)):
    img = read_image_file(await file.read())
    masked_img = predict_mask(model, img)
    return Response(get_image_bytes(masked_img), media_type="image/png")

