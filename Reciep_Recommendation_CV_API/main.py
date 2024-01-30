from fastapi import FastAPI, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
from torchvision.transforms import functional as F
from ultralytics import YOLO
import uvicorn

app = FastAPI()

@app.post("/ingredient_detection")
async def ingredient_detection(input_file: UploadFile):

    # Save the uploaded image temporarily
    image_bytes = await input_file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.save("temp_image.jpg")

    # Load the YOLOv5 model
    try:
        model = YOLO("best_1.pt")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Model loading failed")

    # Perform object detection
    try:
        results = model("temp_image.jpg")[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Object detection failed")

    # Extract ingredient labels from the results
    list_of_ingredients = []
    if results:
        for r in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, cls = r
            cls = int(cls)
            if cls in results[0].names:
                label = results[0].names[cls]
                list_of_ingredients.append(label)
    else:
        print("No results found.")
        # You can return a specific message or handle as needed

    return {"Detected Ingredients": list_of_ingredients}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
