from fastapi import FastAPI, File, UploadFile, Response
from PIL import Image
from io import BytesIO

app = FastAPI()

def read_image_file(file) -> Image.Image:
    img = Image.open(BytesIO(file)).convert("L").convert("RGB").resize((224,224))
    return img
    
def get_image_bytes(img):
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    img = read_image_file(await file.read())
    return Response(get_image_bytes(img), media_type="image/png")

