import pandas
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.utils.detection_predict import YoloInference, MTCCNInference
from src.utils.ocr_predict import EasyOCRModel, EasyOCRCustom


class Inference:

    def __init__(self, detect_model: str = 'yolo', ocr_model: str = 'EasyOCR', type_inf: str = 'demonstration'):
        """
            detect_model (str): yolo / mtcnn / frcnn
            ocr_model (str): EasyOCR / EasyOCR_custom / lpr_custom
            type_inf (str): demonstration / production
        """

        if detect_model == 'yolo':
            self.detect_model = YoloInference()
        else:
            self.detect_model = MTCCNInference()


        if ocr_model == 'EasyOCR':
            self.ocr_model = EasyOCRModel()
        elif ocr_model == 'EasyOCR_custom':
            self.ocr_model = EasyOCRCustom()
        else:
            pass

        self.type_inf = type_inf

    def _get_number(self, image: Image, detect_results: pandas):
        self.xmin = detect_results[0]
        self.ymin = detect_results[1]
        self.xmax = detect_results[2]
        self.ymax = detect_results[3]

        return np.asarray(image.crop((self.xmin, self.ymin, self.xmax, self.ymax)))

    def _demonstration(self, image: Image, text: str) -> Image:
        myFont = ImageFont.truetype('FreeMono.ttf', 30)
        draw = ImageDraw.Draw(image)

        draw.line((self.xmin, self.ymin, self.xmax, self.ymin), fill=128, width=4)
        draw.line((self.xmax, self.ymin, self.xmax, self.ymax), fill=128, width=4)
        draw.line((self.xmax, self.ymax, self.xmin, self.ymax), fill=128, width=4)
        draw.line((self.xmin, self.ymax, self.xmin, self.ymin), fill=128, width=4)
        draw.text((self.xmin, self.ymax), text, fill=(255, 128, 0), font=myFont)

        return {'image': image, 'text_recognition': text}

    def __call__(self, path_to_image: str):
        """
            path_to_image (str): path to image
        """
        image_orig = Image.open(path_to_image)

        detect_results = self.detect_model(image_orig)

        if detect_results is not None:
            try:
                img_number = self._get_number(image_orig, detect_results)
            except Exception:
                return 'Не удалось найти номер автомобиля.'

            try:
                text_recognition = self.ocr_model(img_number)

                if self.type_inf == 'demonstration':
                    if text_recognition is None:
                        text_recognition = 'Не считано.'

                    return self._demonstration(image_orig, text_recognition)
                else:
                    if text_recognition is not None:
                        return text_recognition
                    else:
                        return 'Не удалось считать номер автомобиля.'

            except Exception:
                return 'Не удалось найти номер автомобиля.'

        else:
            return 'Не удалось найти номер автомобиля.'
