import modal

# Define the Modal app and environment
app = modal.App(name="retinaface-service")

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libsm6", "libxext6")  # Install system dependencies
    .run_commands("python -m pip install --upgrade pip")  # Upgrade pip
    .pip_install(
        "fastapi",
        "uvicorn",
        "retina-face",
        "opencv-python-headless",
        "numpy",
        "tf-keras",  # Added tf-keras for TensorFlow compatibility
    )
)

def create_fastapi_app():
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import FileResponse
    import numpy as np
    import cv2
    from retinaface import RetinaFace
    import os
    import uuid

    app = FastAPI()

    @app.post("/detect/")
    async def detect(file: UploadFile = File(...)):
        import traceback
        try:
            print(f"File name: {file.filename}")
            print(f"Content type: {file.content_type}")
            file_size = len(await file.read())
            print(f"File size: {file_size}")
            await file.seek(0)  # Reset file pointer after reading

            # Validate file type
            valid_image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
            valid_video_extensions = [".mp4", ".avi", ".mov", ".mkv"]

            ext = os.path.splitext(file.filename)[-1].lower()
            if not file.content_type or (ext not in valid_image_extensions + valid_video_extensions):
                raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image or video.")

            contents = await file.read()
            np_arr = np.frombuffer(contents, np.uint8)

            if ext in valid_image_extensions:
                # Process as image
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise HTTPException(status_code=400, detail="Invalid image data.")

                # Perform face detection and blur faces
                results = RetinaFace.detect_faces(img)
                if results:
                    for key, value in results.items():
                        x1, y1, x2, y2 = list(map(int, value["facial_area"]))
                        face_region = img[y1:y2, x1:x2]
                        blurred_face = cv2.GaussianBlur(face_region, (51, 51), 30)
                        img[y1:y2, x1:x2] = blurred_face
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Save processed image and return
                output_path = f"processed_{uuid.uuid4().hex}.jpg"
                cv2.imwrite(output_path, img)
                return FileResponse(output_path, media_type="image/jpeg", filename=os.path.basename(output_path))

            elif ext in valid_video_extensions:
                # Process as video
                temp_video_path = "temp_video.mp4"
                with open(temp_video_path, "wb") as f:
                    f.write(contents)

                cap = cv2.VideoCapture(temp_video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 FPS if invalid
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

                print(f"Video FPS: {fps}, Width: {width}, Height: {height}")

                output_path = f"processed_{uuid.uuid4().hex}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = 0

                # Inside the video processing loop
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("No more frames to read.")
                        break

                    # Debug frame properties
                    print(f"Frame dimensions: {frame.shape if ret else 'N/A'}, dtype: {frame.dtype if ret else 'N/A'}")

                    # Ensure frame consistency
                    if frame.shape[1] != width or frame.shape[0] != height:
                        print("Resizing frame to match VideoWriter dimensions.")
                        frame = cv2.resize(frame, (width, height))

                    if frame.dtype != np.uint8:
                        print("Converting frame to uint8.")
                        frame = frame.astype(np.uint8)

                    # Perform face detection and blurring
                    results = RetinaFace.detect_faces(frame)
                    if results:
                        for key, value in results.items():
                            x1, y1, x2, y2 = list(map(int, value["facial_area"]))
                            face_region = frame[y1:y2, x1:x2]
                            blurred_face = cv2.GaussianBlur(face_region, (51, 51), 30)
                            frame[y1:y2, x1:x2] = blurred_face
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    out.write(frame)

                cap.release()
                out.release()
                print(f"Video processing completed. Saved to: {output_path}")
            return FileResponse(output_path, media_type="video/mp4", filename=os.path.basename(output_path))

        except Exception as e:
            print("Error:", e)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Expose FastAPI app as HTTP endpoint
@app.function(image=image, gpu="A10G", timeout=600)  # Increase timeout if needed
@modal.asgi_app()
def fastapi_modal_app():
    return create_fastapi_app()
