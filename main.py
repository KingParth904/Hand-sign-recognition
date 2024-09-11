from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = FastAPI()

# Load your trained model
model = load_model('path_to_your_model.h5')  # Replace with your model's path

# Define the shape of your input image
image_height, image_width, channels = 28, 28, 1  # Adjust based on your model

# Define the actions (labels) your model predicts
actions = ['Hello', 'Morning', 'Afternoon', 'Woman', 'Child', 'Thankyou', 'Study', 'Good']  # Replace with your actual actions

def preprocess_image(image):
    # Convert the image to grayscale if necessary
    if channels == 1 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to the target shape
    image = cv2.resize(image, (image_height, image_width))
    
    # Reshape and normalize the image
    image = image.reshape(1, image_height, image_width, channels)
    image = image.astype('float32') / 255.0
    
    return image

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        # Read the image from the request
        image_data = await image.read()
        np_img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Preprocess the image
        processed_image = preprocess_image(img)

        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = actions[np.argmax(prediction)]

        # Return the prediction as a JSON response
        return JSONResponse(content={'prediction': predicted_class})
    
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
