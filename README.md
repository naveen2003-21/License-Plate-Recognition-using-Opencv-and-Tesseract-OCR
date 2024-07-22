# License-Plate-Recognition-using-Opencv-and-Tesseract-OCR

"Develop a robust license plate recognition system using OpenCV and Tesseract OCR to accurately detect, extract, and interpret license plate information from images or video footage, enabling efficient vehicle identification and verification in real-time applications."

## Requirements

- Python 3.x for project development.
- Essential Python packages: pandas,tensorflow, ultralytics, supervision, opencv-python, pytesseract.

## Features

- License Plate Detection: Utilize computer vision techniques in OpenCV to identify and locate license plates within images or video frames.
- Character Segmentation: Implement algorithms to segment individual characters from the detected license plate for accurate recognition.
- Optical Character Recognition (OCR): Employ Tesseract OCR to convert segmented characters into machine-readable text.
- Accuracy Enhancement: Apply image preprocessing techniques to enhance accuracy, such as noise reduction, image normalization, and contrast adjustments.
- Real-time Processing: Develop the system to process video footage in real-time, enabling live identification of license plates from a continuous video stream.
- Database Integration: Integrate a database to store recognized license plate data for further analysis or record-keeping.
- Permitted Vehicles Verification: Compare recognized license plate numbers against a predefined list of permitted vehicles to validate entry or access.
- User Interface: Create a user-friendly interface to interact with the system, allowing users to input images or video and view recognized license plate details.
- Alerts or Notifications: Implement notifications or alerts for unauthorized vehicles or specific events, providing timely information to designated users or systems.
- Scalability and Robustness: Design the system to be scalable and robust, capable of handling varying environmental conditions, lighting, and vehicle angles for reliable performance.

## Design Steps

### Step 1: Setup and Installation
- Install necessary libraries such as OpenCV, Tesseract OCR, and any additional required packages.
- Import the required libraries into the Jupyter Notebook/Colab environment.
### Step 2: Image/Video Acquisition
- Load an image or capture video footage that contains vehicles and license plates for processing.
### Step 3: Preprocessing
- Use OpenCV for preprocessing tasks such as resizing, grayscale conversion, noise reduction, and edge detection to prepare the image/video for license plate detection.
### Step 4: License Plate Detection
- Implement object detection techniques using OpenCV to locate and extract license plate regions within the image/video frames.
### Step 5: Character Segmentation
- Develop algorithms to segment individual characters from the detected license plate regions.
### Step 6: Optical Character Recognition (OCR)
- Utilize Tesseract OCR to perform character recognition on the segmented characters extracted from the license plate.
### Step 7: Post-processing
- Validate and refine the recognized text by applying text processing techniques like filtering, correction, or normalization.
### Step 8: Verification Against Permitted List
- Compare the recognized license plate text with a predefined list of permitted vehicles' license plate numbers.
### Step 9: Display Results
- Display the recognized license plate information along with the verification result (permitted or not permitted) to the user.
### Step 10: Create a User Interface (Optional)
- Develop a user-friendly interface using widgets or graphical elements to facilitate interaction with the project's functionalities.
### Step 11: Documentation and Testing
- Document the project code with comments and explanations for better understanding.
- Test the project with different images/videos to validate its accuracy and reliability.
### Step 12: Deployment and Integration (Optional)
- Deploy the project for real-time usage or integrate it into other systems for automatic verification and monitoring.

### Coding in Jupyter Notebook or Colab:
- Start by importing the required libraries (OpenCV, Tesseract OCR, etc.).
- Implement each step as a separate code block or function in the Jupyter Notebook/Colab.
- Use markdown cells to add descriptive explanations, instructions, or comments for better clarity and understanding.
- Run the code cells sequentially to execute the steps and observe the results at each stage.

## Architecture Diagram

![LPR architecture](https://github.com/abdulwasih2003/License-Plate-Recognition-using-Opencv-and-Tesseract-OCT/assets/91781810/9c985269-3b51-46f7-a1e0-7d49ebf9d0bd)


## Flowchart

![Flowchart](https://github.com/abdulwasih2003/License-Plate-Recognition-using-Opencv-and-Tesseract-OCT/assets/91781810/c1ab753b-3f19-49ba-9795-6877ef92b0e6)

## Sample Input

https://youtu.be/i2HU6AuSqQ8?si=NRsnbtsL_8BIOX_d

## Program

### Vehicle_recognition:
~~~python
import cv2
from ultralytics import YOLO
import supervision as sv
import subprocess
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set the video path
videopath = r'C:\Users\...\.mp4'
outdir = "output/"
tracked = set()

def main():
    # Create a box annotator for bounding box visualization
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    
    # Load the YOLO model
    model = YOLO("yolov8n.pt")

    # Iterate over the video frames
    for result in model.track(source=videopath, stream=True):
        frame = result.orig_img
        
        # Convert YOLO detections to supervision detections
  
        detections = sv.Detections.from_ultralytics(result)
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        # Filter detections based on class IDs
        detections = detections[
            (detections.class_id == 7)
            | (detections.class_id == 2)
            | (detections.class_id == 3)
        ]
        
        # Create labels for annotations
        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for __, _, confidence, class_id, tracker_id in detections
        ]
        
        # Process each bounding box and save the cropped image if necessary
        for box, cls_id in zip(result.boxes, detections.class_id):
            if box.cls.cpu().numpy().astype(int) in [7, 2, 3]:
                if box.id is not None:
                    tracker_id = box.id.cpu().numpy().astype(int)
                    x, y, x1, y1 = (box.xyxy.numpy().astype(int))[0]
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    img_name = str(model.names[cls_id]) + "_" + str(tracker_id) + ".jpg"
                    path = os.path.join(outdir, img_name)
                    if not os.path.exists(path):
                        cv2.imwrite(path, frame[y:y1, x:x1])
                    else:
                        img_old = cv2.imread(path)
                        area_old = img_old.shape[0]
                        area_new = y1 - y
                        if area_new >= area_old:
                            cv2.imwrite(path, frame[y:y1, x:x1])
        
        # Annotate bounding boxes
        box_annotator.annotate(frame, detections)
        frame = box_annotator.annotate(
            scene=frame, detections=detections, labels=labels
        )
        
        # Display the frame with annotations
        cv2.imshow("yolov8", frame)
        
        # Break loop if ESC key is pressed
        if cv2.waitKey(1) == 27:
            break

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
   
    # Run License_Plate_Classifier.ipynb using the %run magic command
    %run License_Plate_Classifier.ipynb
~~~

### License_Plate_Classifier:

~~~python
import cv2
import os
import pytesseract

# Set the path to the Tesseract executable (replace with your path if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

# Path to the folder containing input images
input_folder = 'C:/Users/.../'

# Path to the folder where cropped license plate images will be saved
output_folder = 'C:/Users/.../'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        detect_license_plate(image_path, output_folder)
~~~

### Text_Conversion

~~~python
from PIL import Image
import pytesseract
import os
import csv
import pandas as pd

# Set the correct path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to perform OCR on an image
def image_to_text(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text.strip()  # Remove leading and trailing whitespaces

# Function to process images and compare extracted text with permitted vehicle list
def process_images(folder_path, permitted_vehicles_xlsx):
    text_list = []
    
    # Read permitted vehicles from the Excel file, skipping the first two rows
    permitted_vehicles_df = pd.read_excel(permitted_vehicles_xlsx)

    # Extract license plate numbers from the 'LP' column
    permitted_vehicles = set(permitted_vehicles_df['LP'].astype(str).str.strip())

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            
            # Extract text from the image
            text = image_to_text(image_path)

            # Clean extracted text to match permitted vehicles format
            cleaned_text = ''.join(filter(str.isalnum, text)).upper()  # Remove spaces

            # Check if the cleaned text corresponds to a permitted vehicle
            if cleaned_text in permitted_vehicles:
                print(f"License Plate Number : {cleaned_text} - Permitted Vehicle")
            else:
                print(f"License Plate Number : {cleaned_text} - Not Permitted Vehicle")

            # Append the text to the list
            text_list.append(text)

    return text_list

if __name__ == "__main__":
    # Path to the folder containing images
    folder_path = 'C:/Users/..../'

    # Path to the file containing permitted vehicles
    permitted_vehicles_xlsx = 'C:/Users/..../'

    # Process images and get the text list
    text_list = process_images(folder_path, permitted_vehicles_xlsx)
~~~

## Sample Output

### Vehicle Recognition

![car_ 2](https://github.com/abdulwasih2003/License-Plate-Recognition-using-Opencv-and-Tesseract-OCT/assets/91781810/7fc621fb-3e4a-411f-b7d2-c362ae228391)

### License Plate Recogniton

![license_plate_0](https://github.com/abdulwasih2003/License-Plate-Recognition-using-Opencv-and-Tesseract-OCT/assets/91781810/1abdc00f-bf68-4c66-ac82-0e7cea6200a1)

### Checking with the list

![final out](https://github.com/abdulwasih2003/License-Plate-Recognition-using-Opencv-and-Tesseract-OCT/assets/91781810/f3002e36-ee29-44bb-bb4d-87ab013838cd)

## Conclusion

The integration of OpenCV and Tesseract OCR in developing a license plate recognition system has demonstrated commendable accuracy in identifying and interpreting license plate information from images and videos. Leveraging sophisticated computer vision algorithms enabled precise localization and extraction of license plates, while Tesseract's OCR capabilities facilitated the conversion of segmented characters into text. The system exhibited resilience in varying environmental conditions, ensuring consistent performance across different lighting, angles, and image qualities. By integrating verification against a predefined list of permitted vehicles, the system offers an additional layer of security for access control. To further enhance its capabilities, future improvements in preprocessing, character segmentation, and potential incorporation of machine learning techniques could amplify the system's robustness, making it adaptable across diverse scenarios and extending its practical utility in traffic management, security systems, and related domains. Ultimately, this project lays a strong foundation for an efficient and reliable license plate recognition system with potential applications in real-world settings.
