from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np


class FaceModelFaceDataset(Dataset):
    def __init__(self, root_face, root_faceModel, transform=None):
        self.root_face = root_face
        self.root_faceModel = root_faceModel
        self.transform = transform

        self.face_images = os.listdir(root_face)
        self.faceModel_images = os.listdir(root_faceModel)
        self.length_dataset = max(len(self.face_images), len(self.faceModel_images))  # 1000, 1500
        self.face_len = len(self.face_images)
        self.faceModel_len = len(self.faceModel_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        face_img = self.face_images[index % self.face_len]
        faceModel_img = self.faceModel_images[index % self.faceModel_len]

        face_path = os.path.join(self.root_face, face_img)
        faceModel_path = os.path.join(self.root_faceModel, faceModel_img)

        face_img = np.array(Image.open(face_path).convert("RGB"))
        faceModel_img = np.array(Image.open(faceModel_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=face_img, image0=faceModel_img)
            face_img = augmentations["image"]
            faceModel_img = augmentations["image0"]
        return face_img, faceModel_img
