import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import gender_guesser.detector as gender
import os
import cv2
import re

# Set the Tesseract executable path if it's not in the system PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize image to increase OCR accuracy
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Apply adaptive thresholding
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img


# Function to extract text from an image
def extract_text_from_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")

    img = preprocess_image(image_path)

    # Extract text using pytesseract with English language specified
    text = pytesseract.image_to_string(img, lang='eng')
    return text


# Function to predict gender from a list of names
def predict_gender(names):
    d = gender.Detector()
    gender_predictions = []
    for name in names:
        first_name = name.split()[0]
        gender_prediction = d.get_gender(first_name)
        gender_predictions.append((name, gender_prediction))
    return gender_predictions


# Function to process the extracted text and predict genders
def process_text_and_predict_gender(text):
    # Use regex to find potential names (simple heuristic: two words starting with a capital letter)
    name_pattern = re.compile(r'\b[A-Z][a-z]*\b \b[A-Z][a-z]*\b')
    names = name_pattern.findall(text)
    gender_predictions = predict_gender(names)
    return gender_predictions



def main(image_path):
    # Extract text from image
    text = extract_text_from_image(image_path)
    print("Extracted Text:", text)

    # Processes text and predict gender
    gender_predictions = process_text_and_predict_gender(text)
    for name, gender_prediction in gender_predictions:
        print(f"Name: {name}, Predicted Gender: {gender_prediction}")


# Example usage
if __name__ == "__main__":
    # Replaces this with the actual path to your scanned image
    image_path = r"C:\Users\guber\Downloads\IMG_7142.jpeg"  # Uses raw string or forward slashes
    main(image_path)
