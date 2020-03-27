from argus.model import load_model
from cnd.ocr.argus_model import CRNNModel
from cnd.ocr.transforms import get_transforms
import numpy as np
from cnd.config import OCR_EXPERIMENTS_DIR, CONFIG_PATH, Config
CV_CONFIG = Config(CONFIG_PATH)


import torch


class Predictor:
    def __init__(self, model_path, converter, image_size=(80, 32), device="cpu"):
        self.model = load_model(model_path, device=device)
        self.ocr_image_size = image_size

        self.transform = get_transforms(image_size)

        self.converter = converter

    def predict(self, images):
        if len(images.shape) == 3:
            images = images[None]

        images = torch.stack([self.transform(img) for img in images])
        pred = self.model.predict(images)
        txts = self.model.preds_converter(pred, len(images))[0]

        return txts
