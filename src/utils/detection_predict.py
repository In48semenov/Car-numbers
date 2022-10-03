import torch
import yaml

with open('./src/configs/inference.yaml') as file:
    detect_model_path = yaml.safe_load(file)['detect_model']


class YoloInference:

    def __init__(self):
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=detect_model_path['path_yolo']
        )

    def __call__(self, img: str):
        results = self.model(img)

        return results


class MTCCNInference:

    def __init__(self):
        pass
