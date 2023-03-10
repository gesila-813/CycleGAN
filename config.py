import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/test"
BATCH_SIZE = 24
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 10
LAMBDA_CYCLE = 10
NUM_WORKERS = 12
NUM_EPOCHS = 1000
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "genh.pth"
CHECKPOINT_GEN_Z = "genz.pth"
CHECKPOINT_CRITIC_H = "critich.pth"
CHECKPOINT_CRITIC_Z = "criticz.pth"

transforms = A.Compose(
    [
        A.Resize(width=200, height=200),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)