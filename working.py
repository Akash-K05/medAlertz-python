from roboflow import Roboflow
import cv2
import easyocr
import numpy as np
import os

# Initialize Roboflow
rf = Roboflow(api_key="QXEU69ZtGV5d9DdttRdN")
project = rf.workspace().project("medicine-images")
model = project.version(1).model

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Initialize for English language

# Predict on the image
image_path = "ericifil.jpg"
result = model.predict(image_path, confidence=40, overlap=30).json()

# Load the image
image = cv2.imread(image_path)
original_image = image.copy()

# Create output directory if it doesn't exist
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Extract medicine name from each detection
detection_count = 0
for prediction in result["predictions"]:
    if prediction["class"] == "medicine":  # Assuming the class name is "medicine"
        detection_count += 1
        
        # Get bounding box coordinates
        x = prediction["x"]
        y = prediction["y"]
        width = prediction["width"]
        height = prediction["height"]
        
        # Convert to top-left, bottom-right coordinates
        x1 = int(x - width/2)
        y1 = int(y - height/2)
        x2 = int(x + width/2)
        y2 = int(y + height/2)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Crop the region of interest
        roi = original_image[y1:y2, x1:x2]
        
        # Use EasyOCR for text recognition
        results = reader.readtext(roi)
        
        # Extract text from results
        medicine_name = ""
        for (bbox, text, prob) in results:
            medicine_name += text + " "
        
        print(f"Medicine name: {medicine_name.strip()}")
        
        # Optional: Display confidence score
        if results:
            print(f"Confidence: {results[0][2]:.2f}")
        
        # Save the cropped region instead of displaying it
        roi_path = os.path.join(output_dir, f"ROI_{detection_count}.jpg")
        cv2.imwrite(roi_path, roi)
        print(f"Saved ROI to {roi_path}")

# Save the image with bounding boxes instead of displaying it
output_path = os.path.join(output_dir, "predictions.jpg")
cv2.imwrite(output_path, image)
print(f"Saved annotated image to {output_path}")

print("Processing complete!")