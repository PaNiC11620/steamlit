from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess, decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np


models_dict = {
    "EfficientNetB0": {
        "model": EfficientNetB0(weights="imagenet"),
        "preprocess": efficientnet_preprocess,
        "target_size": (224, 224)
    },
    "MobileNetV2": {
        "model": MobileNetV2(weights="imagenet"),
        "preprocess": mobilenet_preprocess,
        "target_size": (224, 224)
    },
    "ResNet50": {
        "model": ResNet50(weights="imagenet"),
        "preprocess": resnet_preprocess,
        "target_size": (224, 224)
    }
}

def prepare_image(image, target_size=(224, 224), preprocess_func=efficientnet_preprocess):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_func(image)
    return image

def predict(image: Image.Image, model_name="MobileNetV2", top=5):
    model_info = models_dict.get(model_name, models_dict["MobileNetV2"])
    model = model_info["model"]
    preprocess_func = model_info["preprocess"]
    target_size = model_info["target_size"]

    processed = prepare_image(image, target_size, preprocess_func)
    preds = model.predict(processed, verbose=0)
    return decode_predictions(preds, top=top)[0]