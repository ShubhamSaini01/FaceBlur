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

# Function to create the FastAPI app
def create_fastapi_app():
    from fastapi import FastAPI, UploadFile, File, HTTPException
    import numpy as np
    import cv2
    from retinaface import RetinaFace

    app = FastAPI()

    @app.post("/detect/")
    async def detect(file: UploadFile = File(...)):
        import traceback
        import os 
        try:
            print(f"File name: {file.filename}")
            print(f"Content type: {file.content_type}")
            file_size = len(await file.read())
            print(f"File size: {file_size}")
            await file.seek(0)  # Reset file pointer after reading

            # Validate file type using content_type and extension
            valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
            if not file.content_type or not file.content_type.startswith("image/"):
                ext = os.path.splitext(file.filename)[-1].lower()  # Get file extension
                if ext not in valid_extensions:
                    raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

            # Read and decode the image
            contents = await file.read()
            np_arr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image data.")

            # Perform face detection
            results = RetinaFace.detect_faces(img)
            if not results:
                return {"faces": [], "message": "No faces detected in the image."}

            # Format results
            faces = []
            for key, value in results.items():
                face = {
                    "box": list(map(int, value["facial_area"])),
                    "landmarks": {
                        k: [int(coord) for coord in v]
                        for k, v in value.get("landmarks", {}).items()
                    },
                }
                faces.append(face)

            return {"faces": faces}
        except Exception as e:
            # Log full error for debugging
            print("Error:", e)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
    return app

# Expose FastAPI app as HTTP endpoint
@app.function(image=image, gpu="A10G")  # Adjust GPU as needed
@modal.asgi_app()
def fastapi_modal_app():
    # Explicitly create and return the FastAPI app
    return create_fastapi_app()
