import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import yaml

with open('./src/configs/inference.yaml') as file:
    detect_model_path = yaml.safe_load(file)['detect_model']


class YoloInference:

    def __init__(self):
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=detect_model_path['path_yolo']
        )

    def __call__(self, img: Image):
        results = self.model(img)

        if len(results.pandas().xyxy) > 0:
            xmin = results.pandas().xyxy[0]['xmin'].iloc[0]
            ymin = results.pandas().xyxy[0]['ymin'].iloc[0]
            xmax = results.pandas().xyxy[0]['xmax'].iloc[0]
            ymax = results.pandas().xyxy[0]['ymax'].iloc[0]

            return xmin, ymin, xmax, ymax

        else:
            return None


class FasterRCNNInference:

    def __init__(self):
        self.model = torch.load(detect_model_path['path_frcnn'], map_location='cpu')
        self.model.eval()

    def _get_bbox_with_max_score(self, predict):
        max_score = 0
        for value in predict:
            if max_score < value['scores'].cpu().detach().numpy()[0]:
                bbox = value['boxes'].cpu().detach().numpy()[0]

        return bbox[0], bbox[1], bbox[2], bbox[3]

    @torch.no_grad()
    def __call__(self, img: Image):
        img_tensor = transforms.ToTensor()(np.array(img))
        predict = self.model([img_tensor.to('cpu')])
        if len(predict[0]['boxes'].cpu().detach().numpy()) > 0:
            xmin, ymin, xmax, ymax = self._get_bbox_with_max_score(predict)

            return xmin, ymin, xmax, ymax
        else:
            return None


class MTCCNInference:

    def __init__(self):
        pass
