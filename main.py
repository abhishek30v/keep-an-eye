import os
import time
import logging
import cv2 as cv
from collections import deque
from PIL import Image
from twilio.rest import Client  # Twilio API
from ultralytics import YOLO
import math
import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name="get from cloudinary dashboard",
    api_key="get from cloudinary dashboard",
    api_secret="get from cloudinary dashboard"
)

MODEL_DIR = 'best.pt'

# Ensure the logs directory exists
os.makedirs("./logs", exist_ok=True)

logging.basicConfig(
    filename="./logs/log.log", 
    filemode='a', 
    level=logging.INFO, 
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)

# Twilio credentials (replace these with your actual Twilio credentials)
TWILIO_ACCOUNT_SID = 'get from twilio dashboard'
TWILIO_AUTH_TOKEN = 'get from twilio dashboard'
TWILIO_PHONE_NUMBER = 'twilio number'    # Twilio account trail phone number
RECIPIENT_PHONE_NUMBER = 'receipent number'  # Phone number in which the alert should be sent

# GPS coordinates for the camera location 
LATITUDE = '30.392160'
LONGITUDE = ' 79.318633'
location_on_map = f"https://www.google.com/maps?q={LATITUDE.strip()},{LONGITUDE.strip()}"

# Load the YOLO model
model = YOLO('best.pt')

animals_map = {
    'Tiger' : 0,
    'Leopard' : 1,
    'Cheetah' : 2,
    'Elephant' : 3,
    'Monkey' : 4,
    'Deer' : 5,
    'Lion' : 6,
    'Bear' : 7,
    'Pig' : 8,
    'Bull' : 9
}

class_names = ['Tiger','Leopard','Cheetah','Elephant','Monkey','Deer','Lion','Bear','Pig','Bull']

harmful_to_farms = ['Monkey','Deer','Pig','Bull']
harmful_to_humans = ['Tiger','Leopard','Cheetah','Elephant','Lion','Bear']

# Function to send SMS using Twilio
def send_sms(message):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )
    print(f"SMS Sent: {message.sid}")

# Function to save clip on cloud storage
def upload_clip_and_get_link(buffer, fps, frame_size, timestamp):
    filename = f"detection_{timestamp}.mp4"
    out = cv.VideoWriter(filename, cv.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    for frame in buffer:
        out.write(frame)
    out.release()

    try:
        result = cloudinary.uploader.upload(filename, resource_type="video")
        link = result.get('secure_url')
        print(f"[INFO] Uploaded to Cloudinary: {link}")
    except Exception as e:
        print(f"[ERROR] Cloudinary upload failed: {e}")
        link = None

    os.remove(filename)
    return link


def main():
    print("AnimalDetection")
   
    # Mobile camera URL, replace with your camera stream URL
    mobile_camera_url = "http://192.168.31.142:8080/video"  # It contains the stream URL of the camera that has been used.

    # Open the camera stream
    cap = cv.VideoCapture(mobile_camera_url)

    if not cap.isOpened():
        print("Error: Could not access mobile camera.")
        return

    print("Starting live detection...")

    last_alert_time = 0  # To track last SMS sent time
    alert_interval = 30  # Interval in seconds

    fps = int(cap.get(cv.CAP_PROP_FPS))     # It grabs the frame per seconds from the video capture device
    frame_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))    # Gets the height and widht of each frame, that used late to display
    buffer_size = fps * 5  # 5-sec buffer size to store the actual clip of the animal detected.
    frame_buffer = deque(maxlen=buffer_size)  # Actual data structure where the 5-sec video will be stored.( deque because it stores the recent video and will be easy to delete older videos in constant time.)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve frame. Check the IP camera feed URL.")
            break

        frame_buffer.append(frame)  # Add current frame to buffer

        # Perform inference with YOLO model
        results = model(frame, stream=True)

        # Process bounding boxes and display results
        for info in results:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                class_index = int(box.cls[0])

                if confidence >= 85:  # Adjust this accordingly. ( Here I want to be it atleast 85% sure so I do it 85. )
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Display bounding box and class label
                    label = f'{class_names[class_index]} {confidence}%'
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                    # Send SMS alert and save video clip if a specified animal is detected
                    current_time = time.time()
                    clip_saved_flag = 0
                    if class_names[class_index] in harmful_to_humans and (current_time - last_alert_time >= alert_interval or last_alert_time == 0):
                        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                        message = f"Warning......WILD ANIMAL DETECTED.....!!!!!!\n {class_names[class_index]} is detected at {timestamp} nearby, with {confidence}% confidence.\nLocation: {location_on_map}"
                        send_sms(message)
                        logging.info(f"Sent SMS: {message}")
                        clip_saved_flag = 1
                        last_alert_time = current_time

                    elif class_names[class_index] in harmful_to_farms and (current_time - last_alert_time >= alert_interval or last_alert_time == 0):
                        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                        message = f"Warning...........!!!!!!\n {class_names[class_index]} is detected at {timestamp} near your farm, with {confidence}% confidence.\nLocation: {location_on_map}"
                        send_sms(message)
                        logging.info(f"Sent SMS: {message}")
                        clip_saved_flag = 1
                        last_alert_time = current_time

                    if clip_saved_flag:
                        video_link = upload_clip_and_get_link(list(frame_buffer), fps, frame_size, timestamp)     # send the link of the video saved.
                        if video_link:
                            send_sms(f"Watch video clip here: {video_link}")
                        else:
                            logging.warning("Video upload failed. No link to send.")

        # Display the frame with detected objects
        cv.imshow('Animal Detection', frame)

        # Press 'q' to quit the loop
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
