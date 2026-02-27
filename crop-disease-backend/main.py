from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import io

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- CONFIG ----------------
MODEL_PATH = "discriminator.pth"
DEVICE = torch.device("cpu")

# 38 Classes (EXACT ORDER FROM TRAINING)
disease_classes = [
'Apple___Apple_scab',
'Apple___Black_rot',
'Apple___Cedar_apple_rust',
'Apple___healthy',
'Blueberry___healthy',
'Cherry_(including_sour)___Powdery_mildew',
'Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
'Corn_(maize)___Common_rust_',
'Corn_(maize)___Northern_Leaf_Blight',
'Corn_(maize)___healthy',
'Grape___Black_rot',
'Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
'Grape___healthy',
'Orange___Haunglongbing_(Citrus_greening)',
'Peach___Bacterial_spot',
'Peach___healthy',
'Pepper,_bell___Bacterial_spot',
'Pepper,_bell___healthy',
'Potato___Early_blight',
'Potato___Late_blight',
'Potato___healthy',
'Raspberry___healthy',
'Soybean___healthy',
'Squash___Powdery_mildew',
'Strawberry___Leaf_scorch',
'Strawberry___healthy',
'Tomato___Bacterial_spot',
'Tomato___Early_blight',
'Tomato___Late_blight',
'Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot',
'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
'Tomato___Tomato_mosaic_virus',
'Tomato___healthy'
]

num_classes = len(disease_classes)

# ---------------- REALISTIC TIPS ----------------
def get_suggestion(disease):

    if "healthy" in disease.lower():
        return "Plant appears healthy. Maintain proper irrigation, balanced fertilization, and regular monitoring."

    if "bacterial" in disease.lower():
        return "Remove infected leaves immediately. Avoid overhead irrigation. Apply copper-based bactericide."

    if "early_blight" in disease.lower():
        return "Use recommended fungicides such as Mancozeb. Improve air circulation and avoid leaf wetness."

    if "late_blight" in disease.lower():
        return "Apply systemic fungicides immediately. Ensure proper drainage and destroy infected plants."

    if "powdery_mildew" in disease.lower():
        return "Apply sulfur-based fungicide. Reduce humidity and improve sunlight exposure."

    if "rust" in disease.lower():
        return "Remove affected leaves and apply protective fungicides. Maintain proper spacing."

    if "leaf_mold" in disease.lower():
        return "Improve greenhouse ventilation and apply appropriate fungicide."

    if "leaf_spot" in disease.lower():
        return "Use copper fungicide and avoid water splash on leaves."

    if "virus" in disease.lower():
        return "Remove infected plants immediately. Control insect vectors like whiteflies."

    if "mite" in disease.lower():
        return "Apply recommended miticide and monitor regularly."

    if "scab" in disease.lower():
        return "Apply fungicide during early growth stages and prune for better airflow."

    if "black_rot" in disease.lower():
        return "Remove infected plant debris and apply protective fungicides."

    if "citrus_greening" in disease.lower():
        return "Control psyllid insects and remove infected trees to prevent spread."

    return "Consult local agricultural extension officer for proper treatment."

# ---------------- LOAD MODEL ----------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("Model loaded successfully!")

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- API ----------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        disease = disease_classes[predicted.item()]
        confidence_score = round(confidence.item() * 100, 2)

        return {
            "disease": disease,
            "confidence": f"{confidence_score}%",
            "suggestion": get_suggestion(disease)
        }

    except Exception as e:
        return {"error": str(e)}
