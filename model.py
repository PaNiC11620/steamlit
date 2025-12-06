from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np


model = EfficientNetB0(weights="imagenet")

def prepare_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def predict(image: Image.Image):
    processed = prepare_image(image)
    preds = model.predict(processed)
    return decode_predictions(preds, top=3)[0] # повертає 3 найімовірніші класи