import cv2
import requests
import numpy as np
import time
import os

# URL of the deployed service
SERVICE_URL = "https://shubhamsaini01--retinaface-service-fastapi-modal-app.modal.run/detect/"

# Path to the input file (image or video)
FILE_PATH = "/mnt/c/Users/shubs/output.mp4"  # Update this to your input file (image or video)

# Determine if the file is an image or video
def is_video(file_path):
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    return any(file_path.endswith(ext) for ext in video_extensions)

# Benchmark function for image processing
def process_image(file_path):
    print("Starting image processing...")

    # Start timing
    start_time = time.time()

    with open(file_path, "rb") as file:
        print("Sending image to the server...")
        files = {"file": file}
        response = requests.post(SERVICE_URL, files=files)

    if response.status_code == 200:
        print("Response received from the server.")
        result = response.json()
        faces = result.get("faces", [])
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return

    # Load the image
    print("Loading the image...")
    image = cv2.imread(file_path)
    for face in faces:
        box = face["box"]
        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(image.shape[1], box[2]), min(image.shape[0], box[3])
        face_region = image[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face_region, (51, 51), 30)
        image[y1:y2, x1:x2] = blurred_face
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the output image
    output_path = "output_blurred.jpg"
    cv2.imwrite(output_path, image)
    end_time = time.time()

    print(f"Image processing completed in {end_time - start_time:.2f} seconds.")
    print(f"Output saved to {output_path}")
    return end_time - start_time

# Benchmark function for video processing
def process_video(file_path):
    print("Starting video processing...")

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return None

    # Retrieve video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    print(f"Video Properties: FPS={fps}, Resolution={width}x{height}, Total Frames={total_frames}")

    # Initialize video writer
    output_path = "output_blurred_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read.")
            break

        # Convert frame to bytes and send to the server
        print(f"Processing frame {frame_count + 1}/{total_frames}...")
        _, encoded_image = cv2.imencode(".jpg", frame)
        response = requests.post(SERVICE_URL, files={"file": encoded_image.tobytes()})

        if response.status_code == 200:
            result = response.json()
            faces = result.get("faces", [])
            for face in faces:
                box = face["box"]
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
                face_region = frame[y1:y2, x1:x2]
                blurred_face = cv2.GaussianBlur(face_region, (51, 51), 30)
                frame[y1:y2, x1:x2] = blurred_face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    end_time = time.time()

    print(f"Video processing completed in {end_time - start_time:.2f} seconds.")
    print(f"Output saved to {output_path}")
    return end_time - start_time, fps, width, height, total_frames

# Main logic
print(f"Processing file: {FILE_PATH}")
if is_video(FILE_PATH):
    result = process_video(FILE_PATH)
    if result:
        runtime, fps, width, height, total_frames = result
        print(f"Video Benchmark Results:")
        print(f" - Total Runtime: {runtime:.2f} seconds")
        print(f" - FPS: {fps}")
        print(f" - Resolution: {width}x{height}")
        print(f" - Total Frames: {total_frames}")
else:
    runtime = process_image(FILE_PATH)
    print(f"Image Benchmark Results:")
    print(f" - Total Runtime: {runtime:.2f} seconds")


# Output saved to output_blurred_video.mp4
# Video Benchmark Results:
#  - Total Runtime: 389.44 seconds
#  - FPS: 29.97002997002997
#  - Resolution: 1280x720
#  - Total Frames: 90