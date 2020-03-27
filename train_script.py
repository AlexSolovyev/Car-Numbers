import argparse

from torch.utils.data import DataLoader
from argus.callbacks import MonitorCheckpoint, Checkpoint
from cnd.ocr.dataset import OcrDataset
from cnd.ocr.argus_model import CRNNModel
from cnd.config import OCR_EXPERIMENTS_DIR, CONFIG_PATH, Config
#from cnd.ocr.utils import regular, negative
from cnd.ocr.transforms import get_transforms
from cnd.ocr.metrics import StringAccuracy
import string
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
import glob

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument("-en", "--experiment_name", help="Save folder name", required=True)
parser.add_argument("-gpu_i", "--gpu_index", type=str, default="0", help="gpu index")
args = parser.parse_args()

# define experiment path
EXPERIMENT_NAME = args.experiment_name
EXPERIMENT_DIR = OCR_EXPERIMENTS_DIR / EXPERIMENT_NAME

CV_CONFIG = Config(CONFIG_PATH)

DATASET_PATHS = [
    Path(CV_CONFIG.get("data_path"))
]
# CHANGE YOUR BATCH SIZE
BATCH_SIZE = 64
# 400 EPOCH SHOULD BE ENOUGH
NUM_EPOCHS = 400

alphabet = "ABEKMHOPCTYX"
alphabet += "".join([str(i) for i in range(10)])
alphabet += "-"

MODEL_PARAMS = {"nn_module":
                    ("CRNN", { #DEFINE PARAMS OF YOUR MODEL
                        'image_height' : CV_CONFIG.get("ocr_image_size")[0],
                        'number_input_channels' : CV_CONFIG.get("num_input_channels"), 
                        'number_class_symbols' : len(alphabet),
                        'rnn_size' : CV_CONFIG.get("rnn_size"),
                    }),
                "alphabet": alphabet,
                "loss": {"reduction":"mean"},
                "optimizer": ("Adam", {"lr": 0.0001}),
                # CHANGE DEVICE IF YOU USE GPU
                "device": "cpu",
                }

if __name__ == "__main__":
    if EXPERIMENT_DIR.exists():
        print(f"Folder 'EXPERIMENT_DIR' already exists")
    else:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    transforms =  get_transforms(image_size=(80, 32))

    data_path = 'NumBase'

    all_files = glob.glob(data_path + '/*', recursive=True)
    train_files, val_files = train_test_split(all_files, test_size=0.15)
    train_dataset = OcrDataset(train_files, transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    val_dataset = OcrDataset(val_files, transforms)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    model = CRNNModel(MODEL_PARAMS)

    callbacks = [
        MonitorCheckpoint(EXPERIMENT_DIR, monitor="train_loss", max_saves=6),
        #Checkpoint(EXPERIMENT_DIR),
    ]

    metrics = [
        StringAccuracy(),
    ]

    model.fit(
        train_loader,
        val_loader=val_loader,
        max_epochs=NUM_EPOCHS,
        metrics=metrics,
        callbacks=callbacks,
        metrics_on_train=True,
    )
