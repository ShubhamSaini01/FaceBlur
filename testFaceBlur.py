import cv2
import requests
import numpy as np

# URL of the deployed service
SERVICE_URL = "https://shubhamsaini01--retinaface-service-fastapi-modal-app.modal.run/detect/"
IMAGE_PATH = "/home/predx/faceBlur/run.png"

# Step 1: Send the image to the server
with open(IMAGE_PATH, "rb") as file:
    files = {"file": file}
    response = requests.post(SERVICE_URL, files={"file": file})

# Step 2: Parse the response
if response.status_code == 200:
    result = response.json()
    faces = result.get("faces", [])
else:
    print(f"Error: {response.status_code} - {response.text}")
    exit()

# Step 3: Load the original image
image = cv2.imread(IMAGE_PATH)

# Step 4: Blur faces and draw bounding boxes
for face in faces:
    # Get bounding box coordinates
    box = face["box"]  # [x1, y1, x2, y2]
    x1, y1, x2, y2 = box

    # Ensure coordinates are within image bounds
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)

    # Extract the face region
    face_region = image[y1:y2, x1:x2]

    # Apply Gaussian blur to the face region
    blurred_face = cv2.GaussianBlur(face_region, (51, 51), 30)

    # Replace the original face region with the blurred face
    image[y1:y2, x1:x2] = blurred_face

    # Draw a bounding box around the face
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Step 5: Display the blurred image with bounding boxes
cv2.imshow("Blurred Faces", image)
cv2.waitKey(10)
cv2.destroyAllWindows()

# Optional: Save the blurred image
output_path = "output_blurred.jpg"
cv2.imwrite(output_path, image)
print(f"Blurred image saved to {output_path}")
