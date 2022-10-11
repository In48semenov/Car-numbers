import re
import yaml
import numpy as np
from typing import List

# EasyORCModel
import easyocr

# LPRNet
import torch
from ..models import LPRNet
from ..utils.lprnet_util import decode, preprocess

with open("./src/configs/inference.yaml") as file:
    ocr_model_path = yaml.safe_load(file)["ocr_model"]


class EasyOCRModel:
    """
    Класс модели EasyOCR
    """
    def __init__(
        self,
        lang_list: list = ["ru", "en"],
        text_threshold: float = 0.3,
        device=None,
        paragraph: bool = True,
        detail: int = 1,
        decoder: str = "greedy",
    ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        use_gpu = False
        if self.device.type == "cuda":
            use_gpu = True

        self.text_threshold = text_threshold
        self.paragraph = paragraph
        self.detail = detail
        self.decoder = decoder
        self.model = easyocr.Reader(lang_list=lang_list, gpu=use_gpu)

    def _infer(self, image: np) -> List[str]:
        """
        Метод для детекции и распознования текста.
        :image: изображение в формате bytearray
        :type : bytearray
        """

        prediction = self.model.readtext(
            image=image,
            text_threshold=self.text_threshold,
            paragraph=self.paragraph,
            detail=self.detail,
            decoder=self.decoder,
        )

        return prediction

    def __call__(self, image: np) -> str:
        """
        Извлечение текстов из изображения
        Arguments:
            image (np): изображение
        Returns:
            str: Извлеченный текст из изображения
        """

        prediction = self._infer(image)

        if len(prediction):
            prediction = prediction[-1]
            prediction = re.sub(" ", "", prediction[-1])
            return prediction
        else:
            return None


class EasyOCRCustom:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        use_gpu = False
        if self.device.type == "cuda":
            use_gpu = True

        with open("./src/configs/models/custom_easyocr.yaml") as file:
            inference_params = yaml.safe_load(file)

        self.ocr_model = easyocr.Reader(**inference_params, gpu=use_gpu)

    def __call__(self, image: np) -> str:
        prediction = self.ocr_model.recognize(
            image, detail=0, allowlist="012334567890ABEKMHOPCTYX"
        )

        if len(prediction) > 0:
            prediction = re.sub(" ", "", prediction[0])
            return prediction
        else:
            return None


class LPRNETInference:
    def __init__(self, device=None):
        self.chars = [
            "-",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "A",
            "B",
            "C",
            "E",
            "H",
            "K",
            "M",
            "O",
            "P",
            "T",
            "X",
            "Y",
        ]

        self.chars_dict = {char: i for i, char in enumerate(self.chars)}

        self.device = device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        lprnet_save_model_path = ocr_model_path["path_lpr"]

        self.lprnet = LPRNet(class_num=len(self.chars), dropout_rate=0.5)
        self.lprnet.to(self.device)
        self.lprnet.load_state_dict(
            torch.load(
                lprnet_save_model_path, map_location=lambda storage, loc: storage
            )
        )
        self.lprnet.eval()

    def __call__(self, image: np) -> str:
        cv_image = np.array(image)
        cv_image = cv_image[:, :, ::-1].copy()

        data = preprocess(cv_image).to(self.device)

        preds = self.lprnet(data)
        preds = preds.cpu().detach().numpy()  # (1, 68, 18)
        labels, _ = decode(preds, self.chars)

        if len(labels[0]) > 0:
            return labels[0]
        else:
            return None
