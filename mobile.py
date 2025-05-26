from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import uvicorn

app = FastAPI()
model = YOLO("yolov8n-oiv7.pt")

@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    # Read file contents into a numpy array
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    # Perform inference
    results = model.predict(img, verbose=False)
    res = results[0]

    # Extract unique class names
    classes = []
    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]
        if label not in classes:
            classes.append(label)

    return {"classes": classes}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
