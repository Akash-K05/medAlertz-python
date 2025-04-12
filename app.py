from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import cv2
import easyocr
import uuid
import numpy as np
import pandas as pd
import requests
from roboflow import Roboflow
from pydantic import BaseModel
import ast

# ==========================
#  DIRECTORY CONFIGURATION
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
UPLOADS_DIR = os.path.join(OUTPUT_DIR, "uploads")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================
#  LOAD OFFICIAL DRUG DATABASE (CSV)
# ==========================
CSV_FILE_PATH = os.path.join(BASE_DIR, "official_drugs.csv")

try:
    drug_database = pd.read_csv(CSV_FILE_PATH)
    print(f"✅ Loaded CSV file: {CSV_FILE_PATH}")
except FileNotFoundError:
    print(f"❌ Error: CSV file '{CSV_FILE_PATH}' not found. Creating an empty DataFrame.")
    drug_database = pd.DataFrame(columns=["Composition"])  # Empty DataFrame

# ==========================
#  FASTAPI SETUP
# ==========================
app = FastAPI(title="Smart Drug Authentication API")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# Enable CORS (for frontend API calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
#  INITIALIZE MODELS
# ==========================
reader = easyocr.Reader(['en'], gpu=True)

# Initialize Roboflow for Medicine Detection
rf = Roboflow(api_key="QXEU69ZtGV5d9DdttRdN")
project = rf.workspace().project("medicine-images")
model = project.version(1).model

class MedicineUpdate(BaseModel):
    detection_id: str
    new_name: str

@app.get("/")
async def home():
    return {"status": "healthy"}

# ==========================
#  FUNCTION TO FETCH DRUG COMPOSITION FROM GEMINI API
# ==========================
def get_drug_composition(medicine_name: str):
    try:
        api_key = "AIzaSyDtpOtj1RFITvqdc-tYpye4oevxJfk5R_4"  # Replace with your actual API key
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        payload = {
           "contents": [{
        "parts":[{"text": F'''What is the composition of {medicine_name}? Please provide a simple, concise answer. 
                  do not try to explain only create a perfect 
                  python list of compositons without any other words or context only list [].
                  dont mention python or other words keep it strictly list only. 
                  give it as iupac name, seperated by its common medical name with its dosage amount '''}]
        }]

        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(api_url, json=payload, headers=headers)

     
        cleaned_string = response.json().get("candidates")[0].get("content").get("parts")[0].get("text").strip().replace("'''", "").replace("```", "")
       

       
        
        if response.status_code != 200:
            print(f"❌ Error: API returned {response.status_code} - {response.text}")
            return "Unknown"

        response_data = response.json()
        composition_text = str(cleaned_string)
       
        return composition_text if composition_text else "Unknown"

    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {str(e)}")
        return "Unknown"
# ==========================
#  FUNCTION TO FETCH DRUG USES FROM GEMINI API
# ==========================
def get_drug_uses(medicine_name: str):
    try:
        api_key = "AIzaSyDtpOtj1RFITvqdc-tYpye4oevxJfk5R_4"  # Replace with your actual API key
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        payload = {
           "contents": [{
        "parts":[{"text": f"What are the main therapeutic uses and indications of {medicine_name}? Please provide a brief, concise list."}]
        }]
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(api_url, json=payload, headers=headers)
        
        if response.status_code != 200:
            print(f"❌ Error: API returned {response.status_code} - {response.text}")
            return "Not Available"

        response_data = response.json()
        uses = response_data.get("candidates", [{}])[0].get("content").get("parts")[0].get("text")
        
        return uses if uses else "Not Available"

    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {str(e)}")
        return "Not Available"
# ==========================
#  FUNCTION TO FETCH DRUG SIDE EFFECTS FROM GEMINI API
# ==========================
def get_drug_side_effects(medicine_name: str):
    try:
        api_key = "AIzaSyDtpOtj1RFITvqdc-tYpye4oevxJfk5R_4"  # Replace with your actual API key
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        

        payload = {
           "contents": [{
        "parts":[{"text": F"What are the common side effects of {medicine_name}? Please provide a brief list."}]
        }]
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(api_url, json=payload, headers=headers)
        
        if response.status_code != 200:
            print(f"❌ Error: API returned {response.status_code} - {response.text}")
            return "Not Available"

        response_data = response.json()
        side_effects = response_data.get("candidates", [{}])[0].get("content").get("parts")[0].get("text")

        print(side_effects)
        
        return side_effects if side_effects else "Not Available"

    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {str(e)}")
        return "Not Available"

# ==========================
#  FUNCTION TO VERIFY AUTHENTICITY AGAINST CSV DATABASE
# ==========================
def is_authentic_drug(drug_composition: str, medicine_name: str):
    try:
        # Parse the drug composition string into a list
        medication_list = ast.literal_eval(drug_composition)
        
        # Check if the necessary column exists
        if "Composition" not in drug_database.columns:
            print("❌ Error: 'Composition' column not found in CSV file!")
            return False
        
        # Extract all compounds from the medication list
        # Assuming each sublist has compound name at index 0
        compounds = [item[0].lower() for item in medication_list]
        
        # Get official compositions from database
        official_compositions = drug_database["Composition"].astype(str).str.lower().tolist()
        
        # Check if any compound is found in the official database
        found_in_database = False
        for compound in compounds:
            if any(compound in official_comp for official_comp in official_compositions):
                found_in_database = True
                break
        
        # Check if the medicine is banned in India using Gemini API
        try:
            api_key = "AIzaSyDtpOtj1RFITvqdc-tYpye4oevxJfk5R_4"  # Replace with your actual API key
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            
            # Create a prompt to check if the medicine is banned in India
            prompt = f"""
            Is {medicine_name} banned in India? 
            Please respond with only 'Yes' if it is banned or 'No' if it is not banned.
            If you are uncertain, respond with 'No'.
            """
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(api_url, json=payload, headers=headers)
            
            if response.status_code != 200:
                print(f"❌ Error: API returned {response.status_code} - {response.text}")
                return False
                
            response_data = response.json()
            api_response = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip().lower()
            
            print(f"Ban check response for {medicine_name}: {api_response}")
            
            # Check if the medicine is banned - more strict checking
            if "yes" in api_response or "banned" in api_response:
                print(f"❌ Warning: {medicine_name} appears to be banned in India!")
                return False  # Not authentic if banned
            
            # If it's not banned and found in database, it's authentic
            return found_in_database
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {str(e)}")
            return False
        
    except Exception as e:
        print(f"❌ Error checking drug authenticity: {str(e)}")
        return False
# ==========================
#  GET MEDICINE INFO API ENDPOINT
# ==========================
@app.get("/api/get-medicine-info")
async def get_medicine_info(medicine_name: str):
    try:
        composition = get_drug_composition(medicine_name)
        side_effects = get_drug_side_effects(medicine_name)
        uses=get_drug_uses(medicine_name)
        is_authentic = is_authentic_drug(composition, medicine_name)
        
        return {
            "medicine_name": medicine_name,
            "composition": composition,
            "side_effects": side_effects,
            "uses": uses,
            "is_authentic": is_authentic
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error getting medicine information: {str(e)}")

# ==========================
#  MEDICINE DETECTION API
# ==========================
@app.post("/api/detect/")   
async def detect_medicine(file: UploadFile = File(...)):
    try:
        session_id = str(uuid.uuid4())

        # Save uploaded image
        file_ext = file.filename.split(".")[-1]
        file_path = os.path.join(UPLOADS_DIR, f"{session_id}.{file_ext}")

        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load image using OpenCV
        image = cv2.imread(file_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Run Roboflow model for medicine detection
        result = model.predict(image, confidence=40, overlap=30).json()

        # Initialize detection result
        detection_result = None
        
        # Process predictions
        for prediction in result["predictions"]:
            if prediction["class"] == "medicine":
                detection_id = f"{session_id}_0"  # We'll just use the first medicine detected

                x1 = int(prediction["x"] - prediction["width"] / 2)
                y1 = int(prediction["y"] - prediction["height"] / 2)
                x2 = int(prediction["x"] + prediction["width"] / 2)
                y2 = int(prediction["y"] + prediction["height"] / 2)

                # Mark detection on image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Extract region of interest for OCR
                roi = image[y1:y2, x1:x2]

                # Perform OCR using GPU-accelerated EasyOCR
                ocr_results = reader.readtext(roi)
                medicine_name = " ".join([text for _, text, _ in ocr_results]) if ocr_results else ""
                confidence = ocr_results[0][2] if ocr_results else 0.0

                if medicine_name:
                    # Get composition and side effects from Gemini API
                    composition = get_drug_composition(medicine_name)
                    side_effects = get_drug_side_effects(medicine_name)
                    uses = get_drug_uses(medicine_name)  
                    
                    # Check authenticity
                    is_authentic = is_authentic_drug(composition,medicine_name)
                    
                    # Create detection result
                    detection_result = {
                        "detection_id": detection_id,
                        "medicine_name": medicine_name.strip(),
                        "confidence": float(confidence),
                        "composition": composition,
                        "uses": uses,
                        "side_effects": side_effects,
                        "is_authentic": is_authentic
                    }
                    
                    # We'll process only the first medicine detected
                    break

        # If no medicine detected
        if detection_result is None:
            detection_result = {
                "detection_id": f"{session_id}_none",
                "medicine_name": "No medicine detected",
                "confidence": 0.0,
                "composition": "Unknown",
                "side_effects": "Not Available",
                "uses": "Not Available",
                "is_authentic": False
            }

        # Save the annotated image
        output_path = os.path.join(RESULTS_DIR, f"{session_id}_result.{file_ext}")
        cv2.imwrite(output_path, image)

        # Construct image URL
        image_url = f"http://127.0.0.1:8080/uploads/{session_id}.{file_ext}"
        
        # Add image URL to the result
        detection_result["image_url"] = image_url

        return detection_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error processing image: {str(e)}")

# ==========================
#  UPDATE MEDICINE NAME API
# ==========================
@app.put("/api/update-medicine-name/")
async def update_medicine_name(update: MedicineUpdate):
    try:
        # Get new medicine information
        new_name = update.new_name
        composition = get_drug_composition(new_name)
        side_effects = get_drug_side_effects(new_name)
        uses = get_drug_uses(new_name)
        
        # Check authenticity with the updated medicine name
        # This is the key part - making sure we're using the correct function
        is_authentic = is_authentic_drug(composition, new_name)
        
        print(f"Medicine update: {new_name}, Authentic: {is_authentic}")  # Debug print
        
        return {
            "success": True,
            "detection_id": update.detection_id,
            "updated_name": new_name,
            "composition": composition,
            "side_effects": side_effects,
            "uses": uses,
            "is_authentic": is_authentic  # Ensure this is being passed correctly
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error updating medicine name: {str(e)}")
# ==========================
#  RUN FASTAPI SERVER
# ==========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000, reload=True)