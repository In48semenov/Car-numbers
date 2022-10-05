import math
import yaml

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

# MTCNN
from ..models import MtcnnNet

with open("./src/configs/inference.yaml") as file:
    detect_model_path = yaml.safe_load(file)["detect_model"]


class YoloInference:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=detect_model_path["path_yolo"]
        )
        self.model.eval().to(self.device)

    def __call__(self, img: Image):
        results = self.model(img)

        if len(results.pandas().xyxy) > 0 and (not results.pandas().xyxy[0].empty):
            xmin = results.pandas().xyxy[0]["xmin"].iloc[0]
            ymin = results.pandas().xyxy[0]["ymin"].iloc[0]
            xmax = results.pandas().xyxy[0]["xmax"].iloc[0]
            ymax = results.pandas().xyxy[0]["ymax"].iloc[0]
            return [[xmin, ymin, xmax, ymax]]
        else:
            return None


class FasterRCNNInference:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = torch.load(detect_model_path["path_frcnn"], map_location="cpu")
        self.model.to(self.device)
        self.model.eval()

    def _get_bbox_with_max_score(self, predict):
        max_score = 0
        for value in predict:
            if max_score < value["scores"].cpu().detach().numpy()[0]:
                bbox = value["boxes"].cpu().detach().numpy()[0]
        return [[bbox[0], bbox[1], bbox[2], bbox[3]]]

    @torch.no_grad()
    def __call__(self, img: Image):
        img_tensor = transforms.ToTensor()(np.array(img))
        predict = self.model([img_tensor.to(self.device)])
        if len(predict[0]["boxes"].cpu().detach().numpy()) > 0:
            bboxs = self._get_bbox_with_max_score(predict)
            return bboxs
        else:
            return None


class MTCNNInference:
    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        pnet_model_path = detect_model_path["path_pnet"]
        onet_model_path = detect_model_path["path_onet"]

        self.model = MtcnnNet(device, pnet_model_path, onet_model_path)

    def __call__(self, img: Image):
        cv_image = np.array(img)
        cv_image = cv_image[:, :, ::-1].copy()
        results = self.model.predict(cv_image)[:, :4]

        if len(results):
            return results
        else:
            return None
