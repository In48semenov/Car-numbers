import yaml
import numpy as np

import cv2
from PIL import Image

import torch
import torchvision

# STNEt
from ..models import STNet
from ..utils.stnet_util import preprocess, convert_image

with open("./src/configs/inference.yaml") as file:
    transform_model_path = yaml.safe_load(file)["transform_model"]


class STNetInference:
    def __init__(self, device=None):
        self.device = device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        stn_save_model_path = transform_model_path["path_stn"]

        self.STN = STNet().to(self.device)
        self.STN.load_state_dict(torch.load(stn_save_model_path, map_location=lambda storage, loc: storage))
        self.STN.eval()

    def __call__(self, image: np) -> str:
        cv_image = np.array(image)
        cv_image = cv_image[:, :, ::-1].copy()

        data = preprocess(cv_image).to(self.device)

        transfer = self.STN(data)
        transformed_input_tensor = transfer.cpu()
        out_grid = torchvision.utils.make_grid(transformed_input_tensor)
        out_grid = convert_image(out_grid.detach())

        # out_grid = cv2.cvtColor(out_grid, cv2.COLOR_BGR2RGB)
        out_grid = Image.fromarray(out_grid)

        return out_grid
