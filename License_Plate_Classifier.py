#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os
import pytesseract


# In[ ]:


# Set the path to the Tesseract executable (replace with your path if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# In[ ]:


# Function to perform license plate detection and save cropped images
def detect_license_plate(image_path, output_folder):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and help with contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use the Canny edge detector to find edges in the image
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    
    # Loop through filtered contours and save cropped license plate images
    for i, cnt in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        license_plate = img[y:y+h, x:x+w]
        
        # Save the cropped license plate image to the output folder
        output_path = os.path.join(output_folder, f'license_plate_{i}.png')
        cv2.imwrite(output_path, license_plate)
    
    # Draw the contours on the original image
    cv2.drawContours(img, filtered_contours, -1, (0, 255, 0), 2)
    
    # Display the original image with contours
    cv2.imshow('Contours', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[ ]:


# Path to the folder containing input images
input_folder = 'C:/Users/Abdul/Music/Mini Project/output'


# In[ ]:


# Path to the folder where cropped license plate images will be saved
output_folder = 'C:/Users/Abdul/Music/Mini Project/output/Cropped'


# In[ ]:


# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


# In[ ]:


# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        detect_license_plate(image_path, output_folder)


# In[ ]:


# Run License_plate.ipynb using subprocess
notebook_file = "Plate_to_text_conversion.ipynb"
subprocess.run(["jupyter", "nbconvert", "--to", "script", notebook_file])

# Run the generated Python script for License_plate.ipynb
script_file = "Plate_to_text_conversion.py"
subprocess.run(["python", script_file])

