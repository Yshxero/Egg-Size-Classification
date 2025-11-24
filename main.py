from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
import numpy as np
import pickle
import os
import cv2
from .egg_features import extract_features_from_image

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Static + Templates
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model safely
MODEL_PATH = os.path.join(BASE_DIR, "egg_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

scaler = model["scaler"]
kmeans = model["kmeans"]
feature_keys = model["feature_keys"]
cluster_to_size = model["cluster_to_size"]

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "could not decode image"}

    feats = extract_features_from_image(img)
    X = np.array([[feats[k] for k in feature_keys]])
    X_scaled = scaler.transform(X)
    cluster = int(kmeans.predict(X_scaled)[0])
    size = cluster_to_size.get(cluster, "Unknown")

    return {"cluster": cluster, "size": size}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
