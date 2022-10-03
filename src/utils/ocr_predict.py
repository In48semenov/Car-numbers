import re
from typing import List
import easyocr
import numpy as np
import yaml


class EasyOCRModel:
    """
    Класс модели EasyOCR
    """

    def __init__(self, lang_list: list = ['ru', 'en'],
                 text_threshold: float = 0.3,
                 gpu: bool = False,
                 paragraph: bool = True,
                 detail: int = 1,
                 decoder: str = 'greedy'):

        self.text_threshold = text_threshold
        self.paragraph = paragraph
        self.detail = detail
        self.decoder = decoder
        self.model = easyocr.Reader(lang_list=lang_list, gpu=gpu)

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
            decoder=self.decoder
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

        prediction = self._infer(image)[0]

        if len(prediction) > 0:
            prediction = re.sub(' ', '', prediction[-1])
            return prediction
        else:
            return None


class EasyOCRCustom:

    def __init__(self):

        with open('./src/configs/models/custom_easyocr.yaml') as file:
            inference_params = yaml.safe_load(file)

        self.ocr_model = easyocr.Reader(**inference_params)

    def __call__(self, image: np) -> str:
        prediction = self.ocr_model.recognize(image, detail=0, allowlist="012334567890ABEKMHOPCTYX")

        if len(prediction) > 0:
            prediction = re.sub(' ', '', prediction[0])
            return prediction
        else:
            return None
