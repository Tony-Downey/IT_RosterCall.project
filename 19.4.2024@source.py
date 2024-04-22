import cv2
import numpy as np
import time
import sys
import base64
import os
from keras.models import load_model
from Adafruit_IO import MQTTClient, Client

# Your Adafruit IO credentials
ADAFRUIT_IO_USERNAME = 'Homie_Hung'
ADAFRUIT_IO_KEY = 'aio_jsRH99ZX7Ayn3HIJlvpGrmcTNvZQ'

# Adafruit IO feed names
AIO_FEED_ID = "name"
CONFIDENCE_FEED = "confidence_score"
COUNT_FEED = "people_count"
STATUS_FEED = "status_indicator"
ABSENCE_FEED = "absence"

# Initialize Adafruit IO Client
aio = MQTTClient(ADAFRUIT_IO_USERNAME, ADAFRUIT_IO_KEY)

# Define the folder where the images are saved
image_folder = "D:/Project IT for me/Project for my laptop/images"

# Initialize camera
camera = cv2.VideoCapture(0)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
with open("labels.txt", "r", encoding="utf-8") as file:
    class_names = file.readlines()

# Dictionary to track recognized faces and their confidence scores
recognized_faces = {}

# Variable to count the number of people
people_count = 0

# Variable to keep track of whether to continue detecting faces
continue_detection = True

# Function to capture face using webcam
def capture_face():
    ret, frame = camera.read()
    return frame

# Function to save image to a file
def save_image(image, filename):
    with open(os.path.join(image_folder, filename), 'wb') as file:
        file.write(image)

# Connect callback functions
def connected(client):
    print("Connected to Adafruit IO!")

def disconnected(client):
    print("Disconnected from Adafruit IO!")
    sys.exit(1)

def message(client, feed_id, payload):
    print("Received message from feed {}: {}".format(feed_id, payload))

# Set up Adafruit IO callbacks
aio.on_connect = connected
aio.on_disconnect = disconnected
aio.on_message = message

# Connect to Adafruit IO
aio.connect()

# Function to check for existing images in the folder
def check_existing_images():
    existing_images = []
    try:
        for filename in os.listdir(image_folder):
            if filename.endswith(".jpg"):
                existing_images.append(os.path.join(image_folder, filename))
    except FileNotFoundError:
        pass
    return existing_images

# Variable to store the detected labels
detected_labels = []

while continue_detection:
    # Capture webcam image
    ret, image = camera.read()

    # Display the captured frame
    cv2.imshow('Webcam Feed', image)

    # Wait for a short amount of time to allow OpenCV to process GUI events
    cv2.waitKey(1)

    # Resize image
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Convert image to numpy array and reshape
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize image
    image_normalized = (image_array / 127.5) - 1

    # Predict
    prediction = model.predict(image_normalized)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()

    # Check if face has been recognized before
    if class_name not in recognized_faces:
        recognized_faces[class_name] = True

        # Add detected label to the list
        detected_labels.append(class_name)

        # Increment the people count
        people_count += 1

        # Print prediction
        print("Class:", class_name)

        # Check for black color in the image
        black_color_detected = np.all(image_resized == [0, 0, 0], axis=-1).any()

        # Determine status indicator value based on confidence score
        status_indicator = 1 if prediction[0][index] > 0.7 else 0

        # Generate unique filename based on current timestamp
        timestamp = int(time.time())
        image_filename = "captured_face_{}.jpg".format(people_count)

        # Save the resized face image to a file
        save_image(cv2.imencode('.jpg', image_resized)[1], image_filename)

        # Open the captured face image
        with open(os.path.join(image_folder, image_filename), "rb") as file:
            # Read the image data
            image_data = file.read()

        # Encode the image data as base64
        encoded_image = base64.b64encode(image_data)

        # Convert to string
        image_str = encoded_image.decode('utf-8')

        # Publish data to Adafruit IO feeds
        try:
            aio.publish(AIO_FEED_ID, class_name)
            aio.publish(CONFIDENCE_FEED, str(int(round(prediction[0][index] * 100))))
            aio.publish(COUNT_FEED, str(people_count))
            aio.publish(STATUS_FEED, str(status_indicator))
            aio.publish("image", image_str)
            print("Data uploaded successfully!")
        except Exception as e:
            print("Adafruit IO request failed:", e)

        # Ask if user wants to detect more faces
        while True:
            choice = input("Do you want to detect more faces? (yes/no): ").lower()
            if choice == 'no':
                continue_detection = False
                break
            elif choice == 'yes':
                break
            else:
                print("Invalid choice. Please enter 'yes' or 'no'.")

# Publish missing labels if any after the user chooses not to detect more faces
if not continue_detection:
    missing_labels = ', '.join([label.strip() for label in class_names if label.strip() not in detected_labels])
    try:
        aio.publish(ABSENCE_FEED, missing_labels)
    except Exception as e:
        print("Failed to publish missing labels:", e)

# Release camera and close windows
camera.release()
cv2.destroyAllWindows()
