from fastapi import FastAPI, UploadFile
import torch
from PIL import Image
from io import BytesIO
from torchvision.transforms import functional as F
from ultralytics import YOLO
import uvicorn

#authtoken when we run this on colab
# ngrok.set_auth_token("2UKtYINTgMazz0tJqJajnT70HWZ_5frzbJuUMaUPSJzwPtKiy")

app = FastAPI()

@app.post("/ingredient_detection")
async def ingredient_detection(input_file: UploadFile):

    # Save the uploaded image temporarily
    image_bytes = await input_file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.save("temp_image.jpg")

    # Load the YOLOv5 model
    model = YOLO("best.pt")

    # Perform object detection
    results = model("temp_image.jpg")[0]

    # Extract ingredient labels from the results
    list_of_ingredients=[]
    for r in results[0].boxes.data.tolist():
        x1,y1,x2,y2,score,cls=r
        cls=int(cls)
        if cls in results[0].names:
          label=results[0].names[cls]
          list_of_ingredients.append(label)
    print(list_of_ingredients)

    return {"Detected Ingredients": list_of_ingredients}


if __name__ == "__main__":
    uvicorn.run(app,port=8000)